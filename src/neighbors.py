import numpy as np
import ase
import ase.io
import ase.neighborlist
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################
def get_HOO_bonds(atoms, cutoff = 0.6):
    """
    Generate a list of hydrogen bonds and their corresponding oxygen atoms.

    Args:
    --------
    atoms : ase.Atoms
        The atomic configuration, including positions and cell information.
    cutoff : float, optional
        Maximum distance (in A) considered for identifying bonds, default is 0.6.
    
    Returns:
    --------
    np.ndarray
        Array of hydrogen-oxygen-oxygen (HOO) bond indices, where each row contains
        [H_index, O1_index, O2_index].
    """
    
    # Identify the indices of oxygen (O) and hydrogen (H) atoms
    O_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'O']
    H_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'H']
    
    # Number of O and H atoms
    num_O = len(O_indices)
    num_H = len(H_indices)
    if (num_O * 2) != num_H:
        raise ValueError("Number of O atoms is not twice the number of H atoms")
    num_atoms = num_O + num_H

    #**** O-H bonds ****#
    # Set the cutoff distance for the neighbor search for all atoms
    cutoff_list = [cutoff] * num_atoms
    
    # Create a NeighborList object to find O-H bonds with the cutoff distance
    nl = ase.neighborlist.NeighborList(cutoff_list, 
                                          self_interaction=False, 
                                          bothways=True,
                                          )
    nl.update(atoms)

    # List to store O-H bond pairs
    HOO_bonds = []
    for H_idx in H_indices:
        # Get the neighbors of each hydrogen atom
        neighbors = nl.get_neighbors(H_idx)[0]
        O_neighbors = [idx for idx in neighbors if idx in O_indices]
        HOO_bonds.append([H_idx, *tuple(O_neighbors)])
        
        # Check if there are more than two oxygen atoms around a hydrogen atom
        if len(O_neighbors) > 2:
            raise ValueError("Hydrogen atom has more than two O atoms")
    HOO_bonds = np.array(HOO_bonds)

    return HOO_bonds

####################################################################################################
def get_minimum_bond_indices(positions, box_lengths, HOO_bonds):

    # Extract the positions of hydrogen and oxygen atoms involved in the bonds
    positions_H  = positions[HOO_bonds[:, 0]]
    positions_O1 = positions[HOO_bonds[:, 1]]
    positions_O2 = positions[HOO_bonds[:, 2]]   

    #**** O-H distances ****#
    # Calculate the vector differences between hydrogen and oxygen
    delta_OH1 = positions_H - positions_O1
    delta_OH1 -= np.round(delta_OH1 / box_lengths) * box_lengths 
    distances_OH1 = np.linalg.norm(delta_OH1, axis=1)

    delta_OH2 = positions_H - positions_O2
    delta_OH2 -= np.round(delta_OH2 / box_lengths) * box_lengths
    distances_OH2 = np.linalg.norm(delta_OH2, axis=1)

    # Choose the minimum of the two O-H distances for each hydrogen atom
    minimum_indices = (distances_OH1 <= distances_OH2)

    HOO_bonds_new = np.zeros_like(HOO_bonds)
    for i, row in enumerate(HOO_bonds):
        if minimum_indices[i]:
            HOO_bonds_new[i] = [row[0], row[1], row[2]]
        else:
            HOO_bonds_new[i] = [row[0], row[2], row[1]]
    
    HOO_bonds = np.array(HOO_bonds_new)

    return HOO_bonds

####################################################################################################
def compute_minimum_image(delta, box_lengths):
    """
    Apply the minimum image convention for periodic boundary conditions.
    """
    return delta - jnp.round(delta / box_lengths) * box_lengths

def calculate_distances(positions1, positions2, box_lengths):
    """
    Calculate distances between two sets of positions considering periodic boundaries.
    """
    delta = compute_minimum_image(positions1 - positions2, box_lengths)
    return jnp.linalg.norm(delta, axis=1), delta

def get_HOO_distances(positions, box_lengths, HOO_bonds):
    """
    Calculate O-O and O-H distances for a given set of hydrogen-oxygen-oxygen bonds.
    """
    
    # Extract the positions of hydrogen and oxygen atoms involved in the bonds
    positions_H  = positions[HOO_bonds[:, 0]]
    positions_O1 = positions[HOO_bonds[:, 1]]
    positions_O2 = positions[HOO_bonds[:, 2]]   

    #**** O-H distances ****#
    # Calculate the vector differences between hydrogen and oxygen
    # O-H distances
    d_OH1, delta_OH1 = calculate_distances(positions_H, positions_O1, box_lengths)
    d_OH2, delta_OH2 = calculate_distances(positions_H, positions_O2, box_lengths)

    # O-O distances
    d_OO, delta_OO = calculate_distances(positions_O1, positions_O2, box_lengths)

    # Projection of O-H onto O-O
    norm_dOO = jnp.linalg.norm(delta_OO, axis=1)
    pd_OH1 = jnp.abs(jnp.sum(delta_OH1 * delta_OO, axis=1) / norm_dOO)
    pd_OH2 = jnp.abs(jnp.sum(delta_OH2 * delta_OO, axis=1) / norm_dOO)

    return d_OO, d_OH1, d_OH2, pd_OH1, pd_OH2

####################################################################################################

def calc_order_parameter_novmap(positions, box_lengths, HOO_bonds):
    """
    Calculate order parameters based on HOO bonds.
    """
    d_OO, d_OH1, d_OH2, pd_OH1, pd_OH2 = get_HOO_distances(positions, box_lengths, HOO_bonds)

    # Calculate mean distances and order parameters
    mean_d_OO = jnp.mean(d_OO)
    mean_d_OH1 = jnp.mean(d_OH1)
    mean_d_OH2 = jnp.mean(d_OH2)
    mean_pd_OH1 = jnp.mean(pd_OH1)
    mean_pd_OH2 = jnp.mean(pd_OH2)

    # Order parameter based on distance differences
    mean_d_order = jnp.mean(0.5 * mean_d_OO - mean_pd_OH1)

    return jnp.array([mean_d_order, mean_d_OO, mean_d_OH1, mean_d_OH2, mean_pd_OH1, mean_pd_OH2])

calc_order_parameters = jax.vmap(calc_order_parameter_novmap, 
                                 in_axes=(0, None, None), 
                                 out_axes=(0)
                                 )




#####################################################################################################
if __name__ == '__main__':
    
    if 0:
        from checkpoints import load_pkl_data, load_txt_data
        ckpt_file = "/home/zq/zqdata/iceX/sg_train_t100/l1x128r4.0_ice08c_n016_p_90.00_t_100.0_lev_20_flw_[10_256_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_006000.pkl"
        # ckpt_file = "/home/zq/zqdata/iceX/sg_train_t100/l1x128r4.0_ice10m_n016_p_100.00_t_100.0_lev_20_flw_[10_256_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_006000.pkl"
        ckpt = load_pkl_data(ckpt_file)
        box_lengths = ckpt['atoms_info']['box_lengths']
        positions_init = ckpt['atoms_info']['positions_init']
        
        load_bond_indices = "/home/zq/zqcodeml/waterice/src/structures/bond_indices/ice08c_n016.txt"
        HOO_bonds = load_txt_data(load_bond_indices)
        HOO_bonds = jnp.array(HOO_bonds, dtype = jnp.int64)

        deltas = calc_order_parameter_novmap(positions_init, box_lengths, HOO_bonds)
        print("Deltas:", deltas)

        num_atoms, dim = positions_init.shape
        deltas = calc_order_parameters(positions_init.reshape(1, num_atoms, dim), box_lengths, HOO_bonds)
        print("Deltas:", deltas)
        
        
        ######
        atomcoords_epoch = ckpt['atomcoords_epoch']
        d1, d2, d3, num_atoms, dim = atomcoords_epoch.shape
        atomcoords = atomcoords_epoch.reshape(d1*d2*d3, num_atoms, dim)
        deltas = calc_order_parameters(atomcoords, box_lengths, HOO_bonds)
        deltas = np.array(deltas)
        print("Deltas:", deltas.mean(axis=0))
    
    
    if 1:
        from checkpoints import load_pkl_data, load_txt_data
        from crystal import create_ice_crystal
        # init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p200.00.vasp"
        
        # init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p60.00.vasp"
        # init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p150.00_v.vasp"
        init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p250.00_v.vasp"
        atoms, box_lengths, positions_init, num_molecules, density  = create_ice_crystal(init_stru_path)
        HOO_bonds = get_HOO_bonds(atoms)
        HOO_bonds = get_minimum_bond_indices(positions_init, box_lengths, HOO_bonds)
        for i in range(len(HOO_bonds)):
            print(f"{HOO_bonds[i][0]}  {HOO_bonds[i][1]}  {HOO_bonds[i][2]}")

        # load_bond_indices = "/home/zq/zqcodeml/waterice/src/structures/bond_indices/ice08c_n016.txt"
        load_bond_indices = "/home/zq/zqcodeml/waterice/src/structures/bond_indices/ice08c_n128.txt"
        HOO_bonds_l = load_txt_data(load_bond_indices)
        print(jnp.allclose(HOO_bonds, HOO_bonds_l))

        delta = calc_order_parameter_novmap(positions_init, box_lengths, HOO_bonds)
        print("Deltas:", delta)


    # init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p300.00.vasp"
    # init_box_path = "/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p90.00.vasp"
    # isotope = "H2O"

    # from crystal import create_ice_crystal
    # _, box_lengths, _, _, _ = create_ice_crystal(init_box_path, isotope = isotope)
    
    # atoms, box_lengths, positions_init, num_molecules, density \
    #         = create_ice_crystal(init_stru_path, isotope = isotope, box_lengths = box_lengths)
    # from checkpoints import load_txt_data
    # load_bond_indices = "/home/zq/zqcodeml/waterice/src/structures/bond_indices/ice08c_n128.txt"
    # HOO_bonds_load = load_txt_data(load_bond_indices)
    # print(np.allclose(HOO_bonds, HOO_bonds_load))
        
    # HOO_bonds = get_HOO_bonds(atoms)
    # HOO_bonds = get_minimum_bond_indices(positions_init, box_lengths, HOO_bonds)
    # for i in range(len(HOO_bonds)):
    #     print(f"{HOO_bonds[i][0]}  {HOO_bonds[i][1]}  {HOO_bonds[i][2]}")

    # ##============================================================##
    # list_p = np.arange(10, 150, 5)
    # list_d = np.zeros_like(list_p, dtype=np.float64)
    
    # for i in range(len(list_p)):
        
    #     p = list_p[i]
    #     stru_file = f"structures/relax_l1x128r4.0/tetra_ice08_n016_p{p:.2f}.vasp"
    #     atoms = ase.io.read(stru_file)
    #     box_lengths = np.diag(np.array(atoms.get_cell()))

    #     HOO_bonds = get_HOO_bonds(atoms)
    #     HOO_bonds = get_minimum_bond_indices(atomcoords, box_lengths, HOO_bonds)
    #     print(HOO_bonds)
    #     atomcoords = atoms.get_positions()
    #     # d_OO, d_OH = get_HOO_distances(atomcoords, box_lengths, HOO_bonds)
    #     # d_order = np.mean(0.5 * d_OO -  d_OH)
    #     d_order, d_OO, d_OH = calc_order_parameter(atomcoords, box_lengths, HOO_bonds)
        
    #     list_d[i] = d_order
    #     print(f"pressure: {p:.2f}    order parameter: {d_order:.6f}")
        
    
    # ##============================================================##
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 4), dpi=300)
    # plt.plot(list_p, list_d,  ".-", linewidth=1.0, label="energy")
    # plt.xlabel(r"pressure (GPa)")
    # plt.ylabel(r"order parameter ($\AA$)")
    # plt.show()
    