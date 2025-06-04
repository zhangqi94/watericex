import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from functools import partial
import time
from importlib.metadata import version

## import units and constants
try:
    from src.units import units
except:
    from units import units
    
####################################################################################################
try:
    from src.tools import jaxtreemap
except:
    from tools import jaxtreemap    

####################################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
from .mcmc import mcmc_mode

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 
                            None, 0,
                            None, 0, 0, None,
                            None, None, None,
                            None,
                            ),
                   static_broadcasted_argnums=(1, 3, 10),
                   )
def sample_stateindices_and_x(keys, 
                              sampler, params_van,
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
    
    state_indices = sampler(params_van, key_state, batch_per_device)
    state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
    
    logp_fn = lambda phoncoords: logp(phoncoords, params_flw, state_indices_expanded, wfreqs)
    phoncoords, accept_rate = mcmc_mode(logp_fn, phoncoords, key_mcmc, mc_steps, mc_stddev, invsqrtw)
    
    atomcoords = trans_Q2R(phoncoords)
    
    return keys, state_indices, phoncoords, atomcoords, accept_rate


####################################################################################################
## for nvt & npt ensemble
def make_loss(beta, index_list, clip_factor, box_lengths, HOO_bonds, clip_pot,
              log_prob, logpsi, logpsi_grad_laplacian, calc_order_parameters, 
              ):

    def observable_and_lossfn(params_van, params_flw, 
                              wfreqs, state_indices, phoncoords, atomcoords,
                              potential_energies, stress_vectors,
                              keys,
                              ):
        
        #========== calculate E_local & E_mean ==========
        batch_per_device, num_modes, _ = phoncoords.shape
        
        logp_states = log_prob(params_van, state_indices)
        state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
        
        grad, laplacian = logpsi_grad_laplacian(phoncoords, params_flw, state_indices_expanded, wfreqs, keys)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        
        # trans the unit from eV to Kelvin
        kinetic_energies = (- 0.5 * (units.hbar_in_eVamuA**2) \
                    * (laplacian + (grad**2).sum(axis=(-2, -1)))).real / units.Kelvin_2_eV
        potential_energies = potential_energies.real / units.Kelvin_2_eV  # in the unit of Kelvin
                
        Eloc = jax.lax.stop_gradient(kinetic_energies + potential_energies) # in the unit of Kelvin
        Floc = jax.lax.stop_gradient(logp_states / beta + Eloc) # in the unit of Kelvin
        print("K.shape:", kinetic_energies.shape)
        print("V.shape:", potential_energies.shape)
        print("Eloc.shape:", Eloc.shape)
        print("logp_states.shape:", logp_states.shape)
        print("Floc.shape:", Floc.shape)
        
        ### pressure and stress are in the unit of K/A^3
        stress = stress_vectors / units.Kelvin_2_eV
        
        dim = box_lengths.shape[0]
        box_volume = jnp.prod(box_lengths)
        pressure_from_k  = 2 * kinetic_energies / (dim * box_volume) ## K/A^3
        pressure_from_e = - stress_vectors[:, 0:3].mean(axis=1) / units.Kelvin_2_eV ## K/A^3
        pressure = pressure_from_k + pressure_from_e
        press_all = jnp.stack([pressure, pressure_from_k, pressure_from_e], axis=-1)
        
        Gloc = Floc + pressure * box_volume
        print("Gloc.shape:", Gloc.shape)
        print("P.shape:", press_all.shape)
        print("T.shape:", stress.shape)

        ### order parameter
        delta = calc_order_parameters(atomcoords, box_lengths, HOO_bonds)
        print("delta.shape:", delta.shape)
        
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
        Ss = -logp_states[selected_idx]
        Fs = Floc[selected_idx]
        Gs = Gloc[selected_idx]
        Ps = press_all[selected_idx, :]
        Ts = stress[selected_idx, :]
        Ds = delta[selected_idx, :]
        print("Ks.shape:", Ks.shape)
        print("Vs.shape:", Vs.shape)
        print("Es.shape:", Es.shape)
        print("Ss.shape:", Ss.shape)
        print("Fs.shape:", Fs.shape)
        print("Gs.shape:", Gs.shape)
        print("Ps.shape:", Ps.shape)
        print("Ts.shape:", Ss.shape)
        print("Ds.shape:", Ds.shape)
        
        #========== accumulate observables ==========
        K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
        S_mean,  S2_mean,  F_mean,  F2_mean,  G_mean,  G2_mean, \
        P_mean,  P2_mean,  T_mean,  T2_mean,  D_mean,  D2_mean,  = \
                    jaxtreemap(lambda x: jax.lax.pmean(x, axis_name="p"), 
                           (Ks.mean(),       (Ks**2).mean(),
                            Vs.mean(),       (Vs**2).mean(),
                            Es.mean(),       (Es**2).mean(),
                            Ss.mean(),       (Ss**2).mean(),
                            Fs.mean(),       (Fs**2).mean(),
                            Gs.mean(),       (Gs**2).mean(),
                            Ps.mean(axis=0), (Ps**2).mean(axis=0),
                            Ts.mean(axis=0), (Ts**2).mean(axis=0),
                            Ds.mean(axis=0), (Ds**2).mean(axis=0),
                            )
                            )
                    
        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "S_mean": S_mean, "S2_mean": S2_mean,
                      "F_mean": F_mean, "F2_mean": F2_mean,
                      "G_mean": G_mean, "G2_mean": G2_mean,
                      "P_mean": P_mean, "P2_mean": P2_mean,
                      "T_mean": T_mean, "T2_mean": T2_mean,
                      "D_mean": D_mean, "D2_mean": D2_mean,
                      }

        #========== calculate classical gradient ==========
        def class_lossfn(params_van):
            logp_states = log_prob(params_van, state_indices)
            
            tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F_mean - clip_factor*tv, F_mean + clip_factor*tv)
            gradF_phi = (logp_states * Floc_clipped).mean()
            class_score = logp_states.mean()
            return gradF_phi, class_score
        
        #========== calculate quantum gradient ==========
        def quant_lossfn(params_flw):
            logpsix = logpsi(phoncoords, params_flw, state_indices_expanded, wfreqs)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
            gradF_theta = 2 * ((logpsix * Eloc_clipped.conj()).real.mean())
            quant_score = 2 * (logpsix.real.mean())
            return gradF_theta, quant_score

        return observable, class_lossfn, quant_lossfn

    return observable_and_lossfn


####################################################################################################
## update classical model and quantum model (params_van and params_flw)
@partial(jax.pmap, axis_name="p",
        in_axes =(0, 0, None, 
                  None, 0, 0, 0, 
                  0, 0, 0,
                  0, 0, 0, 0, 
                  None, None, None, None,
                  ),
        out_axes=(0, 0, None, 0, 
                  0, 0, 0,
                  ),
        static_broadcasted_argnums=(14, 15, 16, 17))
def update_van_flw(params_van, params_flw, opt_state,
                   wfreqs, state_indices, phoncoords, atomcoords, 
                   potential_energies, stress_vectors, keys, 
                   datas_acc, grads_acc, class_score_acc, quant_score_acc, 
                   acc_steps, final_step, observable_and_lossfn, optimizer
                   ):
    
    #========== calculate observables and loss functions ==========
    datas, class_lossfn, quant_lossfn = observable_and_lossfn(params_van, params_flw, 
                                                    wfreqs, state_indices, phoncoords, atomcoords,
                                                    potential_energies, stress_vectors,
                                                    keys,
                                                    )

    grad_params_van, class_score = jax.jacrev(class_lossfn)(params_van)
    grad_params_flw, quant_score = jax.jacrev(quant_lossfn)(params_flw)
    
    grads = {'van': grad_params_van, 'flw': grad_params_flw}
    
    grads, class_score, quant_score = jax.lax.pmean((grads, class_score, quant_score), 
                                                    axis_name="p"
                                                    )
                    
    datas_acc, grads_acc, class_score_acc, quant_score_acc = jaxtreemap(lambda acc, i: acc + i, 
                                            (datas_acc, grads_acc, class_score_acc, quant_score_acc), 
                                            (datas,     grads,     class_score,     quant_score)
                                            )

    #========== update at final step ==========
    if final_step:
        datas_acc, grads_acc, class_score_acc, quant_score_acc = jaxtreemap(lambda acc: acc / acc_steps, 
                                            (datas_acc, grads_acc, class_score_acc, quant_score_acc)
                                            )
                        
        grad_params_van, grad_params_flw = grads_acc['van'], grads_acc['flw']
        
        grad_params_van = jaxtreemap(lambda grad, class_score: 
                                    grad - datas_acc["F_mean"] * class_score,
                                    grad_params_van, class_score_acc
                                    )
        grad_params_flw = jaxtreemap(lambda grad, quant_score: 
                                    grad - datas_acc["E_mean"] * quant_score,
                                    grad_params_flw, quant_score_acc
                                    )

        grads_acc = {'van': grad_params_van, 'flw': grad_params_flw}
        updates, opt_state = optimizer.update(grads_acc, opt_state)
        params = {'van': params_van, 'flw': params_flw}
        params = optax.apply_updates(params, updates)
        params_van, params_flw = params['van'], params['flw']

    return params_van, params_flw, opt_state, datas_acc, \
           grads_acc, class_score_acc, quant_score_acc


####################################################################################################
## calculate means and stds for quantities
def calculate_means_and_stds(data, num_molecules, batch, acc_steps):
    # Define constants
    Kelvin_2_meV = units.Kelvin_2_meV        # Conversion factor from Kelvin to meV
    Kelvin_2_GPa = units.Kelvin_2_GPa        # Conversion factor from K/A^3 to GPa
    num_molecules_inv = 1 / num_molecules    # Inverse of the number of molecules for normalization
    batch_acc_inv = 1 / (batch * acc_steps)  # Inverse of batch times accumulation steps

    # List of variables to process
    thermal_vars  = ["F", "E", "K", "V", "G"]  # Thermodynamic quantities
    entropy_vars  = ["S"]                      # Entropy-related quantities
    pressure_vars = ["P", "T"]                 # Pressure-related quantities
    other_vars    = ["D"]                      # Other values (e.g. order parameters)

    # Calculate mean, standard deviation, and apply unit conversion
    computed_quantities = {}
    
    for var in thermal_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * num_molecules_inv * Kelvin_2_meV, std * num_molecules_inv * Kelvin_2_meV
        computed_quantities[var] = (mean, std)
    
    for var in entropy_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * num_molecules_inv, std * num_molecules_inv
        computed_quantities[var] = (mean, std)

    for var in pressure_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * Kelvin_2_GPa, std * Kelvin_2_GPa
        computed_quantities[var] = (mean, std)
        
    for var in other_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        computed_quantities[var] = (mean, std)

    return computed_quantities










####################################################################################################
# ## stores the current stress and pressure values.
# def store_recent_stress(recent_T_vals, recent_P_vals, T_vals, P_vals, num_recent_vals):
    
#     def update_recent_vals(recent_vals, new_vals, num_recent_vals):
#         if recent_vals is None: 
#             recent_vals = jnp.array([new_vals])
#         else:
#             recent_vals = jnp.concatenate([recent_vals, jnp.array([new_vals])])
#         # Keep only the most recent 'num_recent_vals' entries
#         if recent_vals.shape[0] > num_recent_vals:
#             recent_vals = recent_vals[-num_recent_vals:]
#         return recent_vals
    
#     # Extract stress and pressure from T_vals, P_vals.
#     T, T_std = T_vals
#     P, P_std = P_vals
#     # Update recent_T_vals and recent_P_vals
#     recent_T_vals = update_recent_vals(recent_T_vals, (T, T_std), num_recent_vals)
#     recent_P_vals = update_recent_vals(recent_P_vals, (P, P_std), num_recent_vals)
    
#     return recent_T_vals, recent_P_vals


        #========== accumulate observables ==========
        
        # K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
        # S_mean,  S2_mean,  F_mean,  F2_mean,  G_mean,  G2_mean, \
        # P_mean,  P2_mean,  T_mean,  T2_mean, \
        # D_mean,  D2_mean,  = \
        #             jaxtreemap(lambda x: jax.lax.pmean(x, axis_name="p"), 
        #                     (kinetic_energies.mean(),       (kinetic_energies**2).mean(),
        #                     potential_energies.mean(),      (potential_energies**2).mean(),
        #                     Eloc.mean(),                    (Eloc**2).mean(),
        #                     -logp_states.mean(),            (logp_states**2).mean(),
        #                     Floc.mean(),                    (Floc**2).mean(),
        #                     Gloc.mean(),                    (Gloc**2).mean(),
        #                     press_all.mean(axis=0),         (press_all**2).mean(axis=0),
        #                     stress.mean(axis=0),            (stress**2).mean(axis=0),
        #                     delta.mean(axis=0),             (delta**2).mean(axis=0),
        #                     )
        #                     )
