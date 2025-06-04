import torch
# import torch.multiprocessing as mp
# torch.set_default_dtype(torch.float64)

from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from copy import deepcopy

import ase
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################

def make_mace_calculator(mace_model_path: str,
                         mace_dtype = "float64",
                         mace_device: str = "cuda",
                         enable_cueq: bool = True,
                         ):
    
    calc = MACECalculator(model_paths = mace_model_path,
                          device = mace_device,
                          default_dtype = mace_dtype,
                          enable_cueq = enable_cueq,
                          )
    
    return calc


####################################################################################################
def initialize_mace_model(mace_model_path,
                          mace_batch_size,
                          mace_dtype="float64",
                          mace_device="cuda",
                          ):
    
    # Parse model arguments
    mace_args = argparse.Namespace(default_type = mace_dtype,
                                    model = mace_model_path,
                                    device = mace_device,
                                    )
    # if   mace_args.default_type == "float32":
    #     torch.set_default_dtype(torch.float32)
    # elif mace_args.default_type == "float64":
    #     torch.set_default_dtype(torch.float64)

    # Set the default data type for PyTorch and initialize the computation device
    torch_tools.set_default_dtype(mace_args.default_type)
    device = torch_tools.init_device(mace_args.device)

    # Load the pretrained model and move it to the specified device
    model = torch.load(f=mace_args.model, map_location=mace_args.device)
    if mace_args.device == "cuda":
        model = run_e3nn_to_cueq(deepcopy(model), device=device)

    for param in model.parameters():
        param.requires_grad = False

    atomic_numbers = model.atomic_numbers
    r_max = model.r_max
    try:
        heads = model.heads
    except AttributeError:
        heads = None

    #** Set the model to float32 if requested **#
    if   mace_args.default_type == "float32":
        model = model.to(torch.float32)
        print("====** mace model set to float32! **====")
    elif mace_args.default_type == "float64":
        print("====** mace model set to float64! **====")
    
    model = model.to(mace_args.device)

    ####################################################################################################
    def mace_inference(atoms, 
                       atomcoords,
                       compute_stress=True,
                       ):
        compute_force = compute_stress
        
        # Create a copy of the ASE Atoms object for each batch element
        atomcoords = jax.device_put(atomcoords, jax.devices('cpu')[0])
        
        if   mace_args.default_type == "float32":
            atomcoords = np.array(atomcoords, dtype=np.float32)
        elif mace_args.default_type == "float64":
            atomcoords = np.array(atomcoords, dtype=np.float64)
        
        num_devices, batch_per_device, num_atoms, dim = atomcoords.shape
        total_batch = num_devices * batch_per_device
        atomcoords = atomcoords.reshape(total_batch, num_atoms, dim)
        
        atoms_list = []
        for i in range(total_batch):
            atoms.set_positions(atomcoords[i])
            atoms_list.append(atoms.copy())
        
        #===========================================================================================
        # Convert ASE Atoms objects to model-compatible configurations
        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        z_table = utils.AtomicNumberTable([int(z) for z in atomic_numbers])

        data_set = [data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(r_max), heads=heads
                    )
                    for config in configs
                    ]
        # Create a data loader for batched processing of input data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset = data_set,
            batch_size = mace_batch_size,
            shuffle = False,
            drop_last = False,
        )

        # Initialize lists to store predictions
        energies_list = []
        stresses_list = []
        # forces_collection = []

        # Process each batch
        for batch in data_loader:
            
            batch = batch.to(device)
            #** Set the model to float32 if requested **#
            if mace_args.default_type == "float32":
                for key, value in batch:
                    if isinstance(value, torch.Tensor):
                        if value.dtype == torch.float64:
                            setattr(batch, key, value.float())                            
            elif mace_args.default_type == "float64":
                pass
                
            batch_dict = batch.to_dict()
            output = model(batch_dict, 
                           compute_force =compute_force,
                           compute_stress=compute_stress,
                           )

            energies_list.append(torch_tools.to_numpy(output["energy"]))
            
            if compute_stress:
                stresses_list.append(torch_tools.to_numpy(output["stress"]))
                
                # forces = np.split(
                #     torch_tools.to_numpy(output["forces"]),
                #     indices_or_sections=batch.ptr[1:],
                #     axis=0,
                # )
                # forces_collection.append(forces[:-1])  # drop last as its empty

        # Concatenate and convert predictions to numpy arrays
        energies = np.concatenate(energies_list, axis=0).astype(np.float64)
        
        if compute_stress:
            stresses = np.concatenate(stresses_list, axis=0).astype(np.float64)
            # Flatten forces and ensure consistency with input
            # forces_list = [forces for forces_list in forces_collection for forces in forces_list]
            # forces = np.array(forces_list, dtype=np.float64)

        #===========================================================================================
        energies = energies.reshape(num_devices, batch_per_device)
        # forces = forces.reshape(num_devices, batch_per_device, num_atoms, dim)
        # stresses = stresses.reshape(num_devices, batch_per_device, dim, dim)
        
        if compute_stress:
            stresses = stresses.reshape(num_devices, batch_per_device, dim, dim)
        else:
            stresses = np.zeros((num_devices, batch_per_device, dim, dim))
        
        # Transform stresses to the vectors
        potential_energies = energies
        stress_vectors = np.stack([stresses[..., 0, 0], 
                                   stresses[..., 1, 1], 
                                   stresses[..., 2, 2], 
                                   stresses[..., 0, 1], 
                                   stresses[..., 0, 2], 
                                   stresses[..., 1, 2]], axis=-1
                                  )
        
        return potential_energies, stress_vectors

    return mace_inference


####################################################################################################

if __name__ == '__main__':
    import time
    key = jax.random.key(43)
    isotope = "H2O"
    
    init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p40.00.vasp"
    mace_model_path = "/home/zq/zqcodeml/waterice/src/macemodel/mace_iceX_l1x128r4.0.model"
    mace_batch_size = 16
    # mace_dtype = "float64"
    mace_dtype = "float32"
    mace_device = "cuda"
    batch = 32
    compute_stress = False
    
    # create ice crystal
    from crystal import create_ice_crystal
    atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(init_stru_path, 
                                                                                    isotope = isotope,
                                                                                    )
    print("structure:", init_stru_path)
    print("mace model:", mace_model_path)
    print("box_lengths:", box_lengths)
    print("positions_init.shape (Angstrom):", positions_init.shape)
    print("num_molecules:", num_molecules)
    print("density (kg/m^3):", density)
    print("isotope:", isotope, f" [{atoms.get_masses()[0]}  {atoms.get_masses()[-1]}]")

    mace_inference = initialize_mace_model(mace_model_path, 
                                           mace_batch_size,
                                           mace_dtype,
                                           mace_device,
                                           )


    num_atoms, dim = positions_init.shape
    atomcoords = positions_init + 0.1 * jax.random.uniform(key, (1, batch, num_atoms, dim))
    
    #######################
    t1 = time.time()
    potential_energies, stress_vectors = mace_inference(atoms, 
                                                        atomcoords, 
                                                        compute_stress=False,
                                                        )
    t2 = time.time()
    print("----------------------------------------")
    print("** no stress time (s):", t2 - t1)
    print("----------------------------------------")
    print("potential_energies.shape:", potential_energies.shape, potential_energies)
    print("stress_vectors.shape:", stress_vectors.shape, stress_vectors)

    #######################
    t1 = time.time()
    potential_energies, stress_vectors = mace_inference(atoms, 
                                                        atomcoords, 
                                                        compute_stress=True,
                                                        )
    t2 = time.time()
    print("----------------------------------------")
    print("** compute stress time (s):", t2 - t1)
    print("----------------------------------------")
    print("potential_energies.shape:", potential_energies.shape, potential_energies)
    print("stress_vectors.shape:", stress_vectors.shape, stress_vectors)



######################################################################################################
"""
conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice/src
python3 potentialmace.py
"""


