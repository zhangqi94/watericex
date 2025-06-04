import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from functools import partial
import time
from importlib.metadata import version

####################################################################################################
## import units and constants
try:
    from src.units import units
except:
    from units import units
    
try:
    from src.tools import jaxtreemap
except:
    from tools import jaxtreemap    

try:
    from src.mcmc import mcmc_mode
except:
    from mcmc import mcmc_mode
####################################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
# from .mcmc import mcmc_mode

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 0,
                            None, 0, 0, None,
                            None, None, None,
                            None,
                            ),
                   static_broadcasted_argnums=(2, 9),
                   )
def sample_x_mcmc(keys, state_indices,
                logp, phoncoords, params_flw, wfreqs,
                mc_steps, mc_stddev, index_list,
                trans_Q2R,
                ):
    """
    This function samples the variable `phoncoords` and generates corresponding state indices.
    
    Returns:
    --------
    phoncoords : ndarray, shape=(batch_per_device, num_modes, 1
            The newly sampled phonon coordinates for the given batch.
    atomcoords : ndarray, shape=(batch_per_device, num_atoms, dim)
            The corresponding atom coordinates for the given batch.
    state_indices : ndarray, shape=(batch_per_device, num_modes//indices_group)
            The sampled state indices for the given batch.
    """
    
    batch_per_device, num_modes, _ = phoncoords.shape
    invsqrtw  = 1 / jnp.sqrt(wfreqs)
    keys, key_state, key_mcmc = jax.random.split(keys, 3)
    
    state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
    
    logp_fn = lambda phoncoords: logp(phoncoords, params_flw, state_indices_expanded, wfreqs)
    phoncoords, accept_rate = mcmc_mode(logp_fn, phoncoords, key_mcmc, mc_steps, mc_stddev, invsqrtw)
    
    atomcoords = trans_Q2R(phoncoords)
    
    return keys, state_indices, phoncoords, atomcoords, accept_rate


####################################################################################################
## for nvt & npt ensemble
def make_observable_fn(index_list, logpsi_grad_laplacian, clip_pot):

    def observable_fn(params_flw, wfreqs, state_indices, phoncoords, potential_energies, keys,
                    ):
        
        #========== calculate E_local & E_mean ==========
        batch_per_device, num_modes, _ = phoncoords.shape
        
        state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
        
        grad, laplacian = logpsi_grad_laplacian(phoncoords, params_flw, state_indices_expanded, wfreqs, keys)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        
        # trans the unit from eV to Kelvin
        kinetic_energies = (- 0.5 * (units.hbar_in_eVamuA**2) \
                    * (laplacian + (grad**2).sum(axis=(-2, -1)))).real / units.Kelvin_2_eV
        potential_energies = potential_energies.real / units.Kelvin_2_eV  # in the unit of Kelvin
    
        Eloc = jax.lax.stop_gradient(kinetic_energies + potential_energies) # in the unit of Kelvin
        print("K.shape:", kinetic_energies.shape)
        print("V.shape:", potential_energies.shape)
        print("Eloc.shape:", Eloc.shape)
        
        #========== sort observables ==========   
        #----------------------------------------------------
        # sort and take 98% of the lowest energy
        sorted_idx = jnp.argsort(potential_energies)
        cutoff_idx = int(clip_pot * batch_per_device)
        selected_idx = sorted_idx[:cutoff_idx]
        #----------------------------------------------------
        Ks = kinetic_energies[selected_idx]
        Vs = potential_energies[selected_idx]
        Es = Eloc[selected_idx]
        
        print("Ks.shape:", Ks.shape)
        print("Vs.shape:", Vs.shape)
        print("Es.shape:", Es.shape)
        
        #========== accumulate observables ==========
        K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, = \
                    jaxtreemap(lambda x: jax.lax.pmean(x, axis_name="p"), 
                           (Ks.mean(),       (Ks**2).mean(),
                            Vs.mean(),       (Vs**2).mean(),
                            Es.mean(),       (Es**2).mean(),
                            )
                            )

        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      }


        return observable

    return observable_fn


####################################################################################################
## update classical model and quantum model (params_van and params_flw)
@partial(jax.pmap, axis_name="p",
        in_axes =(0, None, 0, 0,
                  0, 0, 
                  0, None, None, None,
                  ),
        out_axes=(0),
        static_broadcasted_argnums=(7, 8, 9))
def calculate_observable(params_flw, wfreqs, state_indices, phoncoords,
                        potential_energies, keys, 
                        datas_acc, acc_steps, final_step, observable_fn, 
                        ):
    
    #========== calculate observables and loss functions ==========
    datas = observable_fn(params_flw, wfreqs, state_indices, phoncoords, potential_energies, keys,)
    datas_acc = jaxtreemap(lambda acc, i: acc + i, datas_acc, datas)

    #========== update at final step ==========
    if final_step:
        datas_acc = jaxtreemap(lambda acc: acc / acc_steps, datas_acc)

    return datas_acc


####################################################################################################
## calculate means and stds for quantities
def calculate_means_and_stds(data, num_molecules, batch, acc_steps):
    # Define constants
    Kelvin_2_meV = units.Kelvin_2_meV   # Conversion factor from Kelvin to meV
    # Kelvin_2_GPa = units.Kelvin_2_GPa   # Conversion factor from K/A^3 to GPa
    # num_molecules_inv = 1 / num_molecules    # Inverse of the number of atoms for normalization
    batch_acc_inv = 1 / (batch * acc_steps)  # Inverse of batch times accumulation steps

    # List of variables to process
    thermal_vars  = ["E", "K", "V"]  # Thermodynamic quantities

    # Calculate mean, standard deviation, and apply unit conversion
    computed_quantities = {}
    
    for var in thermal_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * Kelvin_2_meV, std * Kelvin_2_meV
        computed_quantities[var] = (mean, std)

    return computed_quantities


####################################################################################################
def get_index_basefre(sequence_length, num_spectral_levels=1):
    """
    Only satisfies indices_group = 1 now!!!
    Get indices of base frequency, i.e.:
        [0 0 0 0 0 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0 0 0 0 0]
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0]
        ...
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [2 0 0 0 0 0 0 0 0 0 0 0] (for num_spectral_levels = 2)
    """

    # First row is always zero
    index_basefre = [np.zeros((1, sequence_length), dtype=np.int64)]
    
    # Add rows for each level up to num_spectral_levels
    for i in range(1, num_spectral_levels + 1):
        index_basefre.append(i * np.eye(sequence_length, dtype=np.int64))
    
    index_basefre = np.concatenate(index_basefre, axis=0)
    
    return jnp.array(index_basefre, dtype=jnp.int64)


####################################################################################################
if __name__ == '__main__':
    num_modes = 12
    num_spectral_levels = 0
    index_basefre = get_index_basefre(num_modes, num_spectral_levels)
    print(index_basefre)
