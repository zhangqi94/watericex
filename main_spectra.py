
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
parser.add_argument("--acc_steps", type=int, default=4, 
                    help="accumulate steps for gradient"
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
parser.add_argument("--num_spectral_levels", type=int, default=1, 
                    help="number of spectral levels"
                    )

parser.add_argument("--hutchinson", type=int, default=1, 
                    help="whether to use hutchinson method for flow"
                    )
parser.add_argument("--clip_pot", type=float, default=1.00, 
                    help="Clip potential energies to avoid instability or unexpected behavior."
                    )

## monte carlo parameters
parser.add_argument("--mc_therm", type=int, default=5, 
                    help="MCMC thermalization steps"
                    )
parser.add_argument("--mc_steps", type=int, default=3000, 
                    help="MCMC update steps"
                    )
parser.add_argument("--mc_stddev", type=float, default=0.1, 
                    help="MCMC standard deviation"
                    )

## epoch parameters
parser.add_argument("--folder", type=str, default="data_tests/", 
                    help="folder to save data"
                    )
parser.add_argument("--input_string", type=str, default="H2O_l1x64r4.0_tetra_n016_p20.0", 
                    help="input string for data"
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
  _         __   __                       _             
 (_)        \ \ / /                      | |            
  _  ___ ___ \ V /    ___ _ __   ___  ___| |_ _ __ __ _ 
 | |/ __/ _ \/   \   / __| '_ \ / _ \/ __| __| '__/ _` |
 | | (_|  __/ /^\ \  \__ \ |_) |  __/ (__| |_| | | (_| |
 |_|\___\___\/   \/  |___/ .__/ \___|\___|\__|_|  \__,_|
                         | |                            
                         |_|                            
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


####################################################################################################
print("\n========== load check point files ==========")
#========== params from ckpt ==========
from src.checkpoints import load_txt_data, load_pkl_data
load_ckpt = args.load_ckpt
ckpt = load_pkl_data(load_ckpt)
args_pkl = ckpt["args"]
print("load pkl data from:\n", load_ckpt)

## water ice structure
dim = 3
init_stru_path = args_pkl.init_stru_path
init_box_path = args_pkl.init_box_path
isotope = args_pkl.isotope
load_bond_indices = args_pkl.load_bond_indices
temperature = args_pkl.temperature
beta = 1 / temperature

## flow paramaters
flow_layers = args_pkl.flow_layers
flow_width = args_pkl.flow_width
flow_depth = args_pkl.flow_depth
# num_levels = args_pkl.num_levels
num_levels = 10

#========== params inputs ==========
hutchinson = (args.hutchinson == 1)
clip_pot = args.clip_pot
## mace model path
mace_model_path = args.mace_model_path
mace_batch_size = args.mace_batch_size
mace_dtype = args.mace_dtype
compute_stress = (args.compute_stress == 1)

## number of spectral levels
num_spectral_levels = args.num_spectral_levels

## monte carlo parameters
mc_therm = args.mc_therm
mc_steps = args.mc_steps
mc_stddev = args.mc_stddev

## epoch parameters
folder = args.folder
input_string = args.input_string

####################################################################################################
print("\n========== Initialize files ==========", flush=True)

mode_str = (f"{input_string}")
van_str = f"_lev_{num_spectral_levels}"
mcmc_str = f"_mc_[{mc_therm}_{mc_steps}_{mc_stddev}]"
bth_str = f"_bth_[{batch}_{acc_steps}]_key_{args.seed}"

path = folder + mode_str + van_str + mcmc_str + bth_str

print("#file path:", path)
if not os.path.isdir(path):
    os.makedirs(path)
    print("#create path: %s" % path)

#========== open file ==========
log_filename = os.path.join(path, "data.txt")
print("#data name: ", log_filename, flush=True)
f = open(log_filename, "w", buffering=1, newline="\n")


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
print("order parameter (|dOH - dOO/2|):", dorders[0, 0])
print("OO distance (Angstrom):", dorders[0, 1])
print("OH distance (Angstrom):", dorders[0, 2:])


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

# get imag frequencies
imag_wfreqs = np.sign(wsquares[dim:]) * np.sqrt(np.abs(wsquares[dim:]))
imag_wfreqs_eV = imag_wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm / units.eV_2_cminv
    

print("number of modes:", num_modes)
print("frequencies^2 sqrt(eV/amu)/A:", wsquares)
print("frequencies (cm^-1):", wfreqs_invcm)
print("frequencies (eV):", wfreqs_eV)

#==== check if the frequencies are the same as the ckpt ====
if load_ckpt is not None and load_ckpt != "None":
    print("load frequencies ...")
    wsquares_ckpt = ckpt['atoms_info']['wsquares']
    imag_wfreqs_ckpt = np.sign(wsquares[dim:]) * np.sqrt(np.abs(wsquares[dim:]))
    imag_wfreqs_eV_ckpt = imag_wfreqs_ckpt * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm / units.eV_2_cminv
    
print("\nrecompute the frequencies of each mode (in meV)")
for ii in range(num_modes):
    print("idx0: %06d    w: %.6f    %.6f    %.6f" 
          %(ii+1, wfreqs_eV[ii]/units.meV_2_eV, 
            imag_wfreqs_eV[ii]/units.meV_2_eV,
            imag_wfreqs_eV_ckpt[ii]/units.meV_2_eV,
            )
          )
    f.write( ("%d    %16f    %16f\n") 
            % (ii+1, 
               wfreqs_eV[ii]/units.meV_2_eV, 
               imag_wfreqs_eV[ii]/units.meV_2_eV
               )
            )
    
f.write(("\n"))


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
print("\n========== Initialize Monte Carlo ==========", flush=True)
from src.tools import shard, replicate, automatic_mcstddev

phoncoords = 0.001 * jax.random.normal(key, (num_devices, batch_per_device, num_modes, 1), dtype=jnp.float64)
keys = jax.random.split(key, num_devices)
phoncoords, keys = shard(phoncoords), shard(keys)
params_van, params_flw = replicate((params_van, params_flw), num_devices)


#========== thermalized ==========
from src.vmc_spectra import get_index_basefre, make_observable_fn, sample_x_mcmc, calculate_observable, calculate_means_and_stds
print("number of spectral levels:", num_spectral_levels)
index_basefre = get_index_basefre(num_modes, num_spectral_levels)
state_indices = jnp.tile(index_basefre[0], (num_devices, batch_per_device, 1))
state_indices = shard(state_indices)

#========== thermalized ==========
for ii in range(1, mc_therm+1):
    t1 = time.time()
    keys, state_indices, phoncoords, atomcoords, accept_rate \
                = sample_x_mcmc(keys, state_indices, logp, phoncoords, params_flw, wfreqs,
                                mc_steps, mc_stddev, index_list, trans_Q2R,
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

#========== observable ==========
observable_fn = make_observable_fn(index_list, logpsi_grad_laplacian, clip_pot)


####################################################################################################
print("\n========== Measuring ==========", flush=True)
t0 = time.time()
print("start training:")
print("    E, K, V (in the unit of meV)")


total_epoch = index_basefre.shape[0]
for ii in range(total_epoch):
    
    tf1 = time.time() # time used per epoch
    
    datas_acc =  replicate({"E_mean": 0., "E2_mean": 0., # local energy
                            "K_mean": 0., "K2_mean": 0., # kinetic energy
                            "V_mean": 0., "V2_mean": 0., # potential energy
                            }, 
                            loc_num_devices,
                            )
    accept_rate_acc = shard(jnp.zeros(loc_num_devices))
    
    state_indices = jnp.tile(index_basefre[ii], (num_devices, batch_per_device, 1))
    state_indices = shard(state_indices)

    dts = 0.0 # time used for sampling
    dtm = 0.0 # time used for MACE inference
    
    for jj in range(1, mc_therm+1):
        ts1 = time.time()
        keys, state_indices, phoncoords, atomcoords, accept_rate \
                = sample_x_mcmc(keys, state_indices, logp, phoncoords, params_flw, wfreqs,
                                mc_steps, mc_stddev, index_list, trans_Q2R,
                                )
        ts2 = time.time()
        accept_rate = jnp.mean(accept_rate)
        print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
                            % (jj, accept_rate, mc_stddev, ts2-ts1), flush=True)
        mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
        dts = dts + (ts2 - ts1)

    for acc in range(acc_steps):
        ## sample state indices and phonon positions
        ts1 = time.time()
        keys, state_indices, phoncoords, atomcoords, accept_rate \
                = sample_x_mcmc(keys, state_indices, logp, phoncoords, params_flw, wfreqs,
                                mc_steps, mc_stddev, index_list, trans_Q2R,
                                )
        ts2 = time.time()
        mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
        dts = dts + (ts2 - ts1)

        accept_rate_acc += accept_rate
        final_step = (acc == (acc_steps-1))
        
        ## calculate potential energies and stress tensors using MACE
        tm1 = time.time()
        potential_energies, stress_vectors = mace_inference(atoms, atomcoords, compute_stress)
        tm2 = time.time()
        dtm = dtm + (tm2 - tm1)
        
        ## calculate observables and loss function
        datas_acc = calculate_observable(params_flw, wfreqs, state_indices, phoncoords,
                                        potential_energies, keys, datas_acc, 
                                        acc_steps, final_step, observable_fn, 
                                        )

    # data in the unit of K/atom or K/A^3
    data = jaxtreemap(lambda x: x[0], datas_acc)
    accept_rate = accept_rate_acc[0] / acc_steps
    # mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
    
    # change the unit from K into meV/atom and GPa
    computed_quantities = calculate_means_and_stds(data, num_molecules, batch, acc_steps)
    E, E_std = computed_quantities["E"] # energy
    K, K_std = computed_quantities["K"] # kinetic energy
    V, V_std = computed_quantities["V"] # potential energy

    ####========== print ==========
    tf2 = time.time() # total time used per epoch
    dtf = tf2 - tf1 # time used in this epoch
    
    print("-"*120)
    print("state_index:", state_indices[0, 0])
    print("idx: %06d  acc: %.4f  dx: %.6f  dt: %.3f %.3f %.3f" 
        % (ii, accept_rate, mc_stddev, dtf, dts, dtm)
        )
    print("E: %.2f (%.2f)  K: %.2f (%.2f)  V: %.2f (%.2f)"
        % (E, E_std, K, K_std, V, V_std,)
        )

    ####========== save txt data ==========
    f.write( ("%6d" + 
              "  %.16f"*6 + 
              "  %.16f"*2 + 
              "\n") 
             % (ii, 
                E, E_std, 
                K, K_std, 
                V, V_std, 
                accept_rate, 
                mc_stddev, 
                )
            )
    
    print("-"*120, flush=True)

####################################################################################################





"""
conda activate jax0500-torch251-mace
cd /home/zq/zqcodeml/waterice
python3 main_spectra.py \
    --input_string "H2O_l1x64r4.0_ice08cubic_n016_p40.00" \
    --load_ckpt "/mnt/ssht02data/iceX/train_temp0_t02/H2O_ice08_n016_p_50.00_t_1.0_lev_1_flw_[8_256_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_006000.pkl"  \
    --batch 1024  --acc_steps 8  --mc_therm 5

"""
