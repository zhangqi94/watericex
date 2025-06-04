
import os
import sys
sys.path.append("..")

import numpy as np
# import jax
# jax.config.update("jax_enable_x64", True)
# import jax.numpy as jnp

import matplotlib.pyplot as plt
import ase
import ase.io

from src.neighbors import calc_order_parameters
from src.checkpoints import load_txt_data, load_pkl_data

####################################################################################################
def process_and_save_atomcoords(ckpt_path: str, save_path: str) -> None:
    """
    Process atom coordinates from checkpoint and save as VASP POSCAR files.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        save_path (str): Directory to save the generated files.
    """
    # Load data from checkpoint
    data = load_pkl_data(ckpt_path)
    atoms_info = data['atoms_info']
    atoms = atoms_info['atoms']

    atomcoords_epoch = data['atomcoords_epoch']
    print(f"atomcoords_epoch shape: {atomcoords_epoch.shape}")

    # Reshape atom coordinates
    num_devices, acc_steps, batch_per_device, num_atoms, dim = atomcoords_epoch.shape
    total_batch = num_devices * batch_per_device * acc_steps
    atomcoords = atomcoords_epoch.reshape(total_batch, num_atoms, dim)
    
    # Process and save each structure
    for i in range(total_batch):
        atoms.set_positions(atomcoords[i])
        
        # Ensure save directory exists
        temp_path = os.path.join(save_path, f"{i:06d}")
        os.makedirs(temp_path, exist_ok=True)
        
        save_file = os.path.join(temp_path, f"POSCAR")
        ase.io.write(save_file, atoms, format='vasp')

####################################################################################################
if __name__ == "__main__":
    # Define paths
    # ckpt_path = "/mnt/ssht02data/iceX/ice_l1x128r4.0_cubicmid_n016_p_50.00_t_1.0_lev_1_flw_[10_256_2]_mc_[3000_0.01]_lr_[0.0_2e-05_0.0_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_003000.pkl"
    # save_path = "strus_from_ckpt/cubicmid_p50/"

    press = 40
    ckpt_path = f"/mnt/ssht02data/iceX/ice_l1x128r4.0_cubicmid_n016_p_{press}.00_t_1.0_lev_1_flw_[10_256_2]_mc_[3000_0.01]_lr_[0.0_2e-05_0.0_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_003000.pkl"
    save_path = f"strus_from_ckpt/cubicmid_p{press}/"
    
    
    
    # Process and save data
    process_and_save_atomcoords(ckpt_path, save_path)

