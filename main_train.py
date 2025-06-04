
####################################################################################################
import os
import sys
import time
from importlib.metadata import version


####################################################################################################
import argparse
parser = argparse.ArgumentParser(description= "finite-temperature simulation for ice X")

## device parameters
parser.add_argument("--jax_mem_frac", type=float, default=0.65, 
                    help="fraction of GPU memory to use for JAX"
                    )
parser.add_argument("--batch", type=int, default=256, 
                    help="batch size for training"
                    )
parser.add_argument("--num_devices", type=int, default=1, 
                    help="number of GPU devices to use for training"
                    )
parser.add_argument("--seed", type=int, default=42, 
                    help="random seed"
                    )
parser.add_argument("--acc_steps", type=int, default=1, 
                    help="accumulate steps for gradient"
                    )

## water ice structure
parser.add_argument("--init_stru_path", type=str, default="src/structures/relax/ice08c_n016_p10.00.vasp", 
                    help="path to initial structure"
                    )
parser.add_argument("--isotope", type=str, default="H2O", 
                    help="isotope for water ice"
                    )
parser.add_argument("--temperature", type=float, default=10.0, 
                    help="temperature for water ice"
                    )
parser.add_argument("--init_box_path", type=str, default=None,
                    help="path to initial box"
                    )
parser.add_argument("--load_bond_indices", type=str, default="src/structures/bond_indices/ice08c_n016.txt", 
                    help="path to bond indices"
                    )

## mace model path
parser.add_argument("--mace_model_path", type=str, default="src/macemodel/mace_iceX_l1x64r4.0.model", 
                    help="path to mace model"
                    )
parser.add_argument("--mace_batch_size", type=int, default=8, 
                    help="batch size for mace model"
                    )
parser.add_argument("--mace_dtype", type=str, default="float64", 
                    help="dtype for mace model"
                    )
parser.add_argument("--compute_stress", type=int, default=1, 
                    help="calculate stress or not"
                    )

## van (psa) paramaters
parser.add_argument("--num_levels", type=int, default=1, 
                    help="number of levels"
                    )

## flow paramaters
parser.add_argument("--flow_layers", type=int, default=6, 
                    help="number of layers in flow"
                    )
parser.add_argument("--flow_width", type=int, default=128, 
                    help="width of layers in flow"
                    )
parser.add_argument("--flow_depth", type=int, default=2, 
                    help="depth of layers in flow"
                    )
parser.add_argument("--hutchinson", type=int, default=1, 
                    help="whether to use hutchinson method for flow"
                    )
parser.add_argument("--clip_pot", type=float, default=1.00, 
                    help="Clip potential energies to avoid instability or unexpected behavior."
                    )

# training parameters
parser.add_argument("--lr_class", type=float, default=1e-4, 
                    help="learning rate classical model"
                    )
parser.add_argument("--lr_quant", type=float, default=1e-4, 
                    help="learning rate quantum model"
                    )
parser.add_argument("--min_lr_class", type=float, default=1e-5, 
                    help="minimum learning rate classical model"
                    )
parser.add_argument("--min_lr_quant", type=float, default=1e-5, 
                    help="minimum learning rate quantum model"
                    )
parser.add_argument("--decay_rate", type=float, default=0.99, 
                    help="decay rate of the learning rate"
                    )
parser.add_argument("--decay_steps", type=int, default=100, 
                    help="decay steps of the learning rate"
                    )
parser.add_argument("--decay_begin", type=int, default=500, 
                    help="epochs to start decay"
                    )
parser.add_argument("--clip_factor", type=float, default=5.0, 
                    help="clip factor"
                    )

## monte carlo parameters
parser.add_argument("--mc_therm", type=int, default=10, 
                    help="MCMC thermalization steps"
                    )
parser.add_argument("--mc_steps", type=int, default=3000, 
                    help="MCMC update steps"
                    )
parser.add_argument("--mc_stddev", type=float, default=0.02, 
                    help="MCMC standard deviation"
                    )

## epoch parameters
parser.add_argument("--folder", type=str, default="data_tests/", 
                    help="folder to save data"
                    )
parser.add_argument("--input_string", type=str, default="H2O_l1x64r4.0_tetra_n016_p20.0", 
                    help="input string for data"
                    )
parser.add_argument("--epoch_finished", type=int, default=0, 
                    help="epoch finished"
                    )
parser.add_argument("--epoch_total", type=int, default=10000, 
                    help="total number of epochs"
                    )
parser.add_argument("--epoch_ckpt", type=int, default=20, 
                    help="checkpoint epochs"
                    )
parser.add_argument("--load_ckpt", type=str, default="None", 
                    help="load checkpoint"
                    )

args = parser.parse_args()


####################################################################################################
print("\n========== Initialize parameters ==========", flush=True)

#========== params for jax ==========
jax_mem_frac = args.jax_mem_frac
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{jax_mem_frac}"
import numpy as np

#========== import torch ==========
import torch
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())

#========== import jax ==========
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

print("python version:", sys.version, flush=True)
print("torch.version:", version("torch"), flush=True)
print("jax.version:", version("jax"), flush=True)
print("optax.version:", version("optax"), flush=True)
print("flax.version:", version("flax"), flush=True)
print("numpy.version:", version("numpy"), flush=True)
jax.print_environment_info()

# http://www.patorjk.com/software/taag/#p=display&f=Doom&t=iceX
print(r"""
  _         __   __   _   _ _____ _____ 
 (_)        \ \ / /  | \ | /  __ \_   _|
  _  ___ ___ \ V /   |  \| | /  \/ | |  
 | |/ __/ _ \/   \   | . ` | |     | |  
 | | (_|  __/ /^\ \  | |\  | \__/\ | |  
 |_|\___\___\/   \/  \_| \_/\____/ \_/  

      """, 
      flush=True
      )


####################################################################################################

## physical constants
from src.units import units
from src.tools import jaxtreemap, jaxrandomkey
seed = args.seed
key = jaxrandomkey(seed)

#========== params devices ==========
batch = args.batch
acc_steps = args.acc_steps
num_devices = args.num_devices
batch_per_device = batch // num_devices
print("total batch:", batch)
print("number of devices:", num_devices)
print("batch per device:", batch_per_device)

devices = jax.devices()
tot_num_devices = jax.device_count()
loc_num_devices = jax.local_device_count()
print("total number of GPU devices:", tot_num_devices)
print("")
for i, device in enumerate(devices):
    print("---- ", i, " ", device.device_kind, " ----", flush=True)

#========== params inputs ==========
## water ice structure
dim = 3
init_stru_path = args.init_stru_path
init_box_path = args.init_box_path
isotope = args.isotope
temperature = args.temperature
beta = 1 / temperature
load_bond_indices = args.load_bond_indices

## mace model path
mace_model_path = args.mace_model_path
mace_batch_size = args.mace_batch_size
mace_dtype = args.mace_dtype
compute_stress = (args.compute_stress == 1)
clip_pot = args.clip_pot

## van (psa) paramaters
num_levels = args.num_levels

## flow paramaters
flow_layers = args.flow_layers
flow_width = args.flow_width
flow_depth = args.flow_depth
hutchinson = (args.hutchinson == 1)

## training parameters
lr_class = args.lr_class
lr_quant = args.lr_quant
min_lr_class = args.min_lr_class
min_lr_quant = args.min_lr_quant
decay_rate = args.decay_rate
decay_steps = args.decay_steps
decay_begin = args.decay_begin
clip_factor = args.clip_factor

## monte carlo parameters
mc_therm = args.mc_therm
mc_steps = args.mc_steps
mc_stddev = args.mc_stddev

## epoch parameters
folder = args.folder
input_string = args.input_string
epoch_finished = args.epoch_finished
epoch_total = args.epoch_total
epoch_ckpt = args.epoch_ckpt
load_ckpt = args.load_ckpt


####################################################################################################
print("\n========== Initialize files ==========", flush=True)

mode_str = (f"{input_string}" + f"_t_{temperature}")
van_str = f"_lev_{num_levels}"
flw_str = f"_flw_[{flow_layers}_{flow_width}_{flow_depth}]"
mcmc_str = f"_mc_[{mc_steps}_{mc_stddev}]"
opt_str = f"_lr_[{lr_class}_{lr_quant}_{min_lr_class}_{min_lr_quant}_{decay_rate}_{decay_steps}]"
bth_str = f"_bth_[{batch}_{acc_steps}]_key_{args.seed}"

path = folder + mode_str + van_str + flw_str + mcmc_str + opt_str + bth_str

print("#file path:", path)
if not os.path.isdir(path):
    os.makedirs(path)
    print("#create path: %s" % path)


####################################################################################################
print("\n========== load check point files ==========")
from src.checkpoints import ckpt_filename, save_pkl_data, load_pkl_data, load_txt_data
if load_ckpt is not None and load_ckpt != "None":
    print("load pkl data from:\n", load_ckpt)
    ckpt = load_pkl_data(load_ckpt)
else:
    print("No checkpoint file found. Start from scratch.")


####################################################################################################
print("\n========== Initialize lattice ==========", flush=True)
from src.crystal import create_ice_crystal_init
if init_box_path is not None and init_box_path != "None":
    _, box_lengths, _, _, _ = create_ice_crystal_init(init_box_path, isotope = isotope)
    print("load initial box from:", init_box_path)
    print("init box lengths (Angstrom):", box_lengths)
else:
    box_lengths = None
    print("No initial box file found. Use default box_lengths.")
    
atoms, box_lengths, positions_init, num_molecules, density \
        = create_ice_crystal_init(init_stru_path, isotope = isotope, box_lengths = box_lengths)
num_atoms = num_molecules * dim
box_lengths = jnp.array(box_lengths, dtype=jnp.float64)
print("load initial structure from:", init_stru_path)
print("box_lengths (Angstrom):", box_lengths)
print("positions_init.shape (Angstrom):", positions_init.shape)
print("num_molecules:", num_molecules)
print("density (kg/m^3):", density)
print("isotope:", isotope, f" [{atoms.get_masses()[0]}  {atoms.get_masses()[-1]}]")

    
####################################################################################################
print("\n========== Initialize neighbors ==========", flush=True)
from src.neighbors import get_HOO_bonds, get_minimum_bond_indices, calc_order_parameters

if load_bond_indices is not None and load_bond_indices != "None":
    HOO_bonds = load_txt_data(load_bond_indices)
    HOO_bonds = jnp.array(HOO_bonds, dtype = jnp.int64)
    print("load HOO bonds from:", load_bond_indices)
else:
    HOO_bonds = get_HOO_bonds(atoms)
    HOO_bonds = jnp.array(HOO_bonds, dtype=jnp.int64)
    HOO_bonds = get_minimum_bond_indices(positions_init, box_lengths, HOO_bonds)
    print("calculate HOO bonds ...")

# print(positions_init)
dorders = calc_order_parameters(positions_init.reshape(1, num_atoms, dim), box_lengths, HOO_bonds)

print("HOO bonds:\n", HOO_bonds)
print("order parameter (with projection) (Angstrom) (|dOH - dOO/2|):", dorders[0, 0])
print("d_OO  d_OH1  d_OH2  p_d_OH1  p_d_OH2:", dorders[0, 1:])


####################################################################################################
print("\n========== Initial coordinate transformations ==========", flush=True)
from src.coordtrans import calculate_hessian_matrix, calculate_Dmat_eigs, make_coordinate_transforms
from src.potentialmace import initialize_mace_model, make_mace_calculator                 

print("load mace model from:", mace_model_path)
calc = make_mace_calculator(mace_model_path = mace_model_path)
Dmat, mass_vec = calculate_hessian_matrix(atoms, calc)
#** clear memory **
torch.cuda.empty_cache()

Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat, mass_vec)
# change wfreqs from sqrt(eV/(A^2*amu)) -> s^-1 -> cm^-1
wfreqs_invcm = wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm
# change wfreqs from cm^-1 -> eV
wfreqs_eV = wfreqs_invcm / units.eV_2_cminv

print("number of modes:", num_modes)
print("frequencies^2 sqrt(eV/amu)/A:", wsquares)
print("frequencies (cm^-1):", wfreqs_invcm)
print("frequencies (eV):", wfreqs_eV)


trans_Q2R_novmap, _  = make_coordinate_transforms(positions_init, 
                                                  box_lengths, 
                                                  Q2Rmat,
                                                  R2Qmat,
                                                  coordinate_type = "phonon"
                                                  )
trans_Q2R = jax.vmap(trans_Q2R_novmap, in_axes=(0), out_axes=(0))


####################################################################################################
print("\n========== Initialize mace model ==========", flush=True)
mace_inference = initialize_mace_model(mace_model_path, 
                                       mace_batch_size,
                                       mace_dtype,
                                       )
potential_energies_init, stress_vectors_init = mace_inference(atoms, 
                                                    positions_init.reshape(1, 1, num_atoms, dim),
                                                    compute_stress=True,
                                                    )
potential_energies_init = potential_energies_init[0, 0] / num_molecules
stress_vectors_init = stress_vectors_init[0, 0] * units.eVangstrom_2_GPa
print("init potential energies (eV/H2O):", potential_energies_init)
print("init stress vectors (Txx, Tyy, Tzz, Txy, Tyz, Tzx) (GPa):")
print("  ({:.6g}  {:.6g}  {:.6g}  {:.6g}  {:.6g}  {:.6g})".format(*stress_vectors_init))


####################################################################################################
print("\n========== Initialize wave functions ==========", flush=True)
from src.orbitals import make_orbitals_1d, logphi_base
fn_wavefunctions, fn_energies = make_orbitals_1d(hbar = units.hbar_in_eVamuA)
print("hbar (eV * amu)^(1/2) * A:", units.hbar_in_eVamuA)

print("zero-point energy & excited state energies (in eV/molecule):")
## ZPE = 0.5 * np.sum(wfreqs_eV) / num_molecules
for i in range(10):
    energy = jnp.sum(fn_energies(jnp.array([i] * num_modes, dtype=jnp.int64), wfreqs))
    print("---- level: %.2d    energy: %.12f ----" % (i, energy/num_molecules), flush=True)


####################################################################################################
print("\n========== Initialize autoregressive model ==========", flush=True)
from src.psa import make_product_spectra_ansatz
seq_length, indices_group = num_modes, 1
van = make_product_spectra_ansatz(num_levels, 
                                  indices_group, 
                                  num_modes,
                                  )

params_van = van.init(key)
raveled_params_van, _ = ravel_pytree(params_van)
print("product state ansatz:  [num_levels: %d,  seq_length: %d]" 
                            % (num_levels, num_modes))
print("    #parameters in the probability table: %d" % raveled_params_van.size, flush=True)

wfreqs_K_init = wfreqs_eV / units.Kelvin_2_eV # change into the unit of Kelvin

#**** load ckpt ****
if load_ckpt is not None and load_ckpt != "None":
    print("load params_van ...", flush=True)
    params_van = ckpt['params_van']
    wfreqs_K_init = ckpt['atoms_info']['wfreqs_K_init']

from src.sampler import make_psa_sampler
sampler, log_prob_novmap, index_list = make_psa_sampler(van, 
                                                        num_levels, 
                                                        seq_length, 
                                                        indices_group, 
                                                        wfreqs_K_init, 
                                                        beta, 
                                                        )
log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)


####################################################################################################
print("\n========== Initialize flow model ==========", flush=True)
from src.flow import make_flow_model
flow = make_flow_model(flow_layers, 
                       flow_width, 
                       flow_depth, 
                       num_modes,
                       )

params_flw = flow.init(key, jnp.zeros((num_modes, 1), dtype=jnp.float64))
raveled_params_flw, _ = ravel_pytree(params_flw)
print("flow model (real nvp) [layers: %d,  hidden layers: %d, %d]" 
                            % (flow_layers, flow_width, flow_depth))
print("    #parameters in the flow model: %d" % raveled_params_flw.size, flush=True)   

#**** load ckpt ****
if load_ckpt is not None and load_ckpt != "None":
    print("load params_flw ...", flush=True)
    params_van = ckpt['params_flw']

#========== logpsi = logphi + 0.5*logjacdet ==========
if hutchinson:
    print("use Hutchinson trick to calculate logpsi")
else:
    print("no Hutchinson trick to calculate logpsi")
from src.logpsi import make_logpsi, make_logphi_logjacdet, make_logp, make_logpsi_grad_laplacian
logpsi_novmap = make_logpsi(flow, fn_wavefunctions, logphi_base)
logphi, logjacdet = make_logphi_logjacdet(flow, fn_wavefunctions, logphi_base)
logp = make_logp(logpsi_novmap)
logpsi, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi_novmap, 
                                                            forloop=True, 
                                                            hutchinson=hutchinson, 
                                                            logphi=logphi, 
                                                            logjacdet=logjacdet
                                                            )


####################################################################################################
print("\n========== Initialize optimizer ==========", flush=True)

lr_schedule_class = optax.exponential_decay(init_value       = lr_class,
                                            transition_steps = decay_steps,
                                            decay_rate       = decay_rate,
                                            transition_begin = decay_begin,
                                            end_value        = min_lr_class,
                                            )
lr_schedule_quant = optax.exponential_decay(init_value       = lr_quant,
                                            transition_steps = decay_steps,
                                            decay_rate       = decay_rate,
                                            transition_begin = decay_begin,
                                            end_value        = min_lr_quant,
                                            )

if epoch_finished > 0:
    current_lr_class = lr_schedule_class(epoch_finished)
    current_lr_quant = lr_schedule_quant(epoch_finished)
    lr_schedule_class = optax.exponential_decay(init_value       = current_lr_class,
                                                transition_steps = decay_steps,
                                                decay_rate       = decay_rate,
                                                transition_begin = 0,
                                                end_value        = min_lr_class,
                                                )
    lr_schedule_quant = optax.exponential_decay(init_value       = current_lr_quant,
                                                transition_steps = decay_steps,
                                                decay_rate       = decay_rate,
                                                transition_begin = 0,
                                                end_value        = min_lr_quant,
                                                )
    print("load learning rate (continue) ...")
    print("    current lr classical: %g    quantum: %g" %(current_lr_class, current_lr_quant), flush=True)

optimizer_class = optax.adam(lr_schedule_class)
optimizer_quant = optax.adam(lr_schedule_quant)

optimizer = optax.multi_transform({'class': optimizer_class, 'quant': optimizer_quant},
                                    param_labels={'van': 'class', 'flw': 'quant'})
params = {'van': params_van, 'flw': params_flw}
opt_state = optimizer.init(params)

print("optimizer adam, learning rate:")
print("    initial classical: %g    quantum: %g" % (lr_class, lr_quant))
print("    minimum classical: %g    quantum: %g" % (min_lr_class, min_lr_quant))
print("    decay rate: %g    decay steps: %d    decay begin: %d" %(decay_rate, decay_steps, decay_begin))


####################################################################################################
print("\n========== Initialize Monte Carlo ==========", flush=True)
from src.tools import shard, replicate, automatic_mcstddev

phoncoords = 0.001 * jax.random.normal(key, (num_devices, batch_per_device, num_modes, 1), dtype=jnp.float64)
keys = jax.random.split(key, num_devices)
phoncoords, keys = shard(phoncoords), shard(keys)
# print("---------------------- shape check ----------------------")
# print("keys shape:", keys.shape, type(keys))
# print("phoncoords shape:", phoncoords.shape, type(phoncoords))
# print("---------------------------------------------------------")
params_van, params_flw = replicate((params_van, params_flw), num_devices)

from src.vmc import sample_stateindices_and_x, make_loss, update_van_flw, calculate_means_and_stds
#========== thermalized ==========
for ii in range(1, mc_therm+1):
    t1 = time.time()
    keys, state_indices, phoncoords, atomcoords, accept_rate \
                = sample_stateindices_and_x(keys, 
                                            sampler, params_van,
                                            logp, phoncoords, params_flw, wfreqs,
                                            mc_steps, mc_stddev, index_list,
                                            trans_Q2R,
                                            )
    t2 = time.time()
    accept_rate = jnp.mean(accept_rate)
    print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
                        % (ii, accept_rate, mc_stddev, t2-t1), flush=True)
    mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)

print("---------------------- shape check ----------------------")
print("keys shape:", keys.shape, type(keys))
print("state_indices shape:", state_indices.shape, type(state_indices))
print("phoncoords shape:", phoncoords.shape, type(phoncoords))
print("atomcoords shape:", atomcoords.shape, type(atomcoords))
print("---------------------------------------------------------", flush=True)

#========== observable, loss function ==========
observable_and_lossfn = make_loss(beta, index_list, clip_factor, box_lengths, HOO_bonds, clip_pot,
                                  log_prob, logpsi, logpsi_grad_laplacian, calc_order_parameters, 
                                  )


#========== open file ==========
log_filename = os.path.join(path, "data.txt")
print("#data name: ", log_filename, flush=True)
f = open(log_filename, 
            "w" if epoch_finished == 0 else "a", 
            buffering=1, 
            newline="\n"
            )

atoms_info =   {"atoms": atoms,
                "num_atoms": num_atoms,
                "num_molecules": num_molecules,
                "dim": dim,
                "density": density,
                "box_lengths": box_lengths, 
                "positions_init": positions_init,
                "Dmat": Dmat,
                "mass_vec": mass_vec,
                "Q2Rmat": Q2Rmat,
                "R2Qmat": R2Qmat,
                "num_modes": num_modes,
                "wsquares": wsquares,
                "wfreqs": wfreqs,
                "wfreqs_invcm": wfreqs_invcm,
                "wfreqs_eV": wfreqs_eV,
                "wfreqs_K_init": wfreqs_K_init,
                "HOO_bonds": HOO_bonds,
                }


####################################################################################################
print("\n========== Training ==========", flush=True)
t0 = time.time()
print("start training:")
print("    F, E, K, V, G (in the unit of meV/H2O)")
print("    S: entropy per molecules (in the unit of kB/H2O)")
print("    P: total, atomic, and electronic pressure, (in the unit of GPa)")
print("    T: stress tensor, (in the unit of GPa)")
print("    D: order parameters, (in the unit of Angstrom)")

for ii in range(epoch_finished + 1, epoch_total+1):
    
    tf1 = time.time() # time used per epoch
    
    datas_acc =  replicate({"F_mean": 0., "F2_mean": 0., # Helmholtz free energy
                            "E_mean": 0., "E2_mean": 0., # local energy
                            "K_mean": 0., "K2_mean": 0., # kinetic energy
                            "V_mean": 0., "V2_mean": 0., # potential energy
                            "S_mean": 0., "S2_mean": 0., # entropy
                            "G_mean": 0., "G2_mean": 0., # Gibbs free energy
                            "P_mean": jnp.zeros(3), "P2_mean": jnp.zeros(3), # pressure
                            "T_mean": jnp.zeros(6), "T2_mean": jnp.zeros(6), # stress
                            "D_mean": jnp.zeros(6), "D2_mean": jnp.zeros(6), # order parameters
                            }, 
                            loc_num_devices
                            )

    grads_acc = shard( jaxtreemap(jnp.zeros_like, {'van': params_van, 'flw': params_flw}))
    class_score_acc = shard(jaxtreemap(jnp.zeros_like, params_van))
    quant_score_acc = shard(jaxtreemap(jnp.zeros_like, params_flw))
    accept_rate_acc = shard(jnp.zeros(loc_num_devices))
    phoncoords_epoch = jnp.zeros((loc_num_devices, acc_steps, batch_per_device, num_modes, 1),   dtype=jnp.float64)
    atomcoords_epoch = jnp.zeros((loc_num_devices, acc_steps, batch_per_device, num_atoms, dim), dtype=jnp.float64)

    dts = 0.0 # time used for sampling
    dtm = 0.0 # time used for MACE inference
    for acc in range(acc_steps):
        ## sample state indices and phonon positions
        ts1 = time.time()
        keys, state_indices, phoncoords, atomcoords, accept_rate \
                    = sample_stateindices_and_x(keys, 
                                                sampler, params_van,
                                                logp, phoncoords, params_flw, wfreqs,
                                                mc_steps, mc_stddev, index_list,
                                                trans_Q2R,
                                                )
        ts2 = time.time()
        dts = dts + (ts2 - ts1)

        accept_rate_acc += accept_rate
        final_step = (acc == (acc_steps-1))
        
        ## calculate potential energies and stress tensors using MACE
        tm1 = time.time()
        potential_energies, stress_vectors = mace_inference(atoms, atomcoords, compute_stress)
        tm2 = time.time()
        dtm = dtm + (tm2 - tm1)
        
        ## calculate observables and loss function
        params_van, params_flw, opt_state, datas_acc, grads_acc, class_score_acc, quant_score_acc \
            = update_van_flw(params_van, params_flw, opt_state,
                    wfreqs, state_indices, phoncoords, atomcoords, potential_energies, stress_vectors, keys, 
                    datas_acc, grads_acc, class_score_acc, quant_score_acc, 
                    acc_steps, final_step, observable_and_lossfn, optimizer
                    )

        phoncoords_epoch = phoncoords_epoch.at[:, acc, :, :, :].set(phoncoords)
        atomcoords_epoch = atomcoords_epoch.at[:, acc, :, :, :].set(atomcoords)

    # data in the unit of K/molecule or K/A^3
    data = jaxtreemap(lambda x: x[0], datas_acc)
    accept_rate = accept_rate_acc[0] / acc_steps
    mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
    # args.mcstddev = mc_stddev
    
    # change the unit from K into meV/molecule and GPa
    computed_quantities = calculate_means_and_stds(data, num_molecules, batch, acc_steps)
    F, F_std = computed_quantities["F"] # Helmholtz free energy
    E, E_std = computed_quantities["E"] # energy
    K, K_std = computed_quantities["K"] # kinetic energy
    V, V_std = computed_quantities["V"] # potential energy
    S, S_std = computed_quantities["S"] # entropy
    G, G_std = computed_quantities["G"] # Gibbs free energy
    P, P_std = computed_quantities["P"] # pressure
    T, T_std = computed_quantities["T"] # stress
    D, D_std = computed_quantities["D"] # order parameters
    
    ####========== print ==========
    tf2 = time.time() # total time used per epoch
    dtf = tf2 - tf1 # time used in this epoch
    
    print("-"*120)
    print("iter: %05d  acc: %.4f  dx: %.6f  dt: %.3f %.3f %.3f" 
        % (ii, accept_rate, mc_stddev, dtf, dts, dtm)
        )
    print("F: %.2f (%.2f)  E: %.2f (%.2f)  K: %.2f (%.2f)  V: %.2f (%.2f)  S: %.6f (%.6f)  G: %.2f (%.2f)"
        % (F, F_std, E, E_std, K, K_std, V, V_std, S, S_std, G, G_std)
        )
    print("P: %.3f %.3f %.3f (%.3f %.3f %.3f)  T: %.3f %.3f %.3f %.3f %.3f %.3f (%.3f %.3f %.3f %.3f %.3f %.3f)"
        % (*tuple(P), *tuple(P_std), *tuple(T), *tuple(T_std))
        )
    print("D: %.6f %.6f %.6f %.6f %.6f %.6f (%.6f %.6f %.6f %.6f %.6f %.6f)"
        % (*tuple(D), *tuple(D_std)), flush=True
        )

    ####========== save txt data ==========
    f.write( ("%6d" + 
              "  %.16f"*12 + 
              "  %.16f"*6 + 
              "  %.16f"*12 + 
              "  %.16f"*12 + 
              "  %.16f"*2 + 
              "\n") 
             % (ii, 
                F, F_std, 
                E, E_std, 
                K, K_std, 
                V, V_std, 
                S, S_std, 
                G, G_std, 
                *tuple(P), *tuple(P_std),
                *tuple(T), *tuple(T_std), 
                *tuple(D), *tuple(D_std),
                accept_rate, 
                mc_stddev, 
                )
            )

    if ii % epoch_ckpt == 0:
        ckpt = {"args": args, 
                "keys": keys,
                "phoncoords_epoch": phoncoords_epoch,
                "atomcoords_epoch": atomcoords_epoch,
                "params_flw": jaxtreemap(lambda x: x[0], params_flw), 
                "params_van": jaxtreemap(lambda x: x[0], params_van),
                "atoms_info": atoms_info,
                }

        save_ckpt_filename = ckpt_filename(ii, path)
        save_pkl_data(ckpt, save_ckpt_filename)
        print("save file: %s" % save_ckpt_filename, flush=True)
        print("total time used: %.3fs (%.3fh),  training speed: %.3f epochs per hour. (%.3fs per step)" 
            % ((tf2-t0), 
               (tf2-t0)/3600, 
                3600.0/(tf2-t0)*ii, 
               (tf2-t0)/ii
                ), 
            flush=True
            )
        print("-"*120)
        jax.print_environment_info()
        
    if ii == epoch_finished + 10:
        print("-"*120)
        jax.print_environment_info()


####################################################################################################

"""
conda activate jax0500-torch240-mace

conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice
python3 main_train.py  --compute_stress 1  --hutchinson 1 \
    --init_stru_path "src/structures/relax/ice08c_n016_p60.00.vasp" \
    --init_box_path "None" \
    --mace_model_path "src/macemodel/mace_iceX_l1x128r4.0.model"  \
    --mace_dtype "float32"
"""