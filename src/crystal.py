import numpy as np
import ase
import ase.io

## import units and constants
try:
    from src.units import units, mass_H_list, mass_O_list
except:
    from units import units, mass_H_list, mass_O_list

####################################################################################################
def create_ice_crystal(stru_path,
                       supercell_size = None,
                       box_lengths = None,
                       isotope = "H2O",
                       ):
    """
    Creates an ice crystal structure from a given file, potentially modifies its size, 
    adjusts the box dimensions, and calculates relevant properties like density.

    Args:
    --------
    stru_path (str): Path to the input structure file (e.g., .xyz, .vasp, .cif).
    supercell_size (list or None, optional): A list of three integers defining the supercell size 
                                          (e.g., [2, 2, 2]) to repeat the unit cell. Defaults to None.
    box_lengths (list or None, optional): A list of three floats specifying the box dimensions 
                                          (e.g., [10.0, 10.0, 10.0]). Defaults to None.
    isotope (str, optional): Specifies the isotope of water (either "H2O" or "D2O"). Defaults to "H2O".

    Returns:
    --------
    atoms (ase.Atoms): Atoms object representing the ice crystal.
    box_lengths (numpy.ndarray): dim=3 representing the box dimensions.
    num_molecules (int): The number of water molecules in the structure.
    density (float): The density of the ice crystal (in kg/m^3).
    """
    
    ## Read structure file and create atoms object
    atoms = ase.io.read(stru_path)
    
    if supercell_size is not None:
        supercell_size = np.array(supercell_size, dtype=np.int64)
        atoms = atoms.repeat(supercell_size)
    # atoms = atoms[atoms.get_atomic_numbers().argsort()]
        
    if box_lengths is not None:
        box_lengths = np.array(box_lengths, dtype=np.float64)
        atoms.set_cell(box_lengths, scale_atoms = True,)
    
    box_lengths = np.diag(np.array(atoms.get_cell()))
    box_volume = np.prod(box_lengths)
    
    ## set initial positions
    positions_init = atoms.get_positions()
    
    ## get number of water molecules
    num_H = atoms.get_chemical_symbols().count("H")
    num_O = atoms.get_chemical_symbols().count("O")
    if (num_O * 2) != num_H:
        raise ValueError("Number of O atoms is not twice the number of H atoms")
    num_molecules = num_O
    
    ## get mass of water (in the unit of amu)
    if   isotope == "H2O":
        mass_H, mass_O = mass_H_list[0], mass_O_list[0]
    elif isotope == "D2O":
        mass_H, mass_O = mass_H_list[1], mass_O_list[0]
    ## set masses of atoms
    atoms.set_masses([mass_H] * num_H + [mass_O] * num_O)
    
    ## density (in the unit of kg/m^3)
    density = (mass_H * num_H + mass_O * num_O) * units.amu_2_kg / (box_volume * (units.angstrom_2_m**3))
    
    return atoms, box_lengths, positions_init, num_molecules, density


####################################################################################################
def create_ice_crystal_init(stru_path,
                            box_lengths = None,
                            isotope = "H2O",
                            ):
    """
    Creates an ice crystal structure from a given file, potentially modifies its size, 
    adjusts the box dimensions, and calculates relevant properties like density.

    Args:
    --------
    stru_path (str): Path to the input structure file (e.g., .xyz, .vasp, .cif).
    box_lengths (list or None, optional): A list of three floats specifying the box dimensions 
                                          (e.g., [10.0, 10.0, 10.0]). Defaults to None.
    isotope (str, optional): Specifies the isotope of water (either "H2O" or "D2O"). Defaults to "H2O".

    Returns:
    --------
    atoms (ase.Atoms): Atoms object representing the ice crystal.
    box_lengths (numpy.ndarray): dim=3 representing the box dimensions.
    num_molecules (int): The number of water molecules in the structure.
    density (float): The density of the ice crystal (in kg/m^3).
    """
    
    ## Read structure file and create atoms object
    atoms = ase.io.read(stru_path)
        
    if box_lengths is not None:
        box_lengths = np.array(box_lengths, dtype=np.float64)
        atoms.set_cell(box_lengths, scale_atoms = True,)
    
    box_lengths = np.diag(np.array(atoms.get_cell()))
    box_volume = np.prod(box_lengths)
    
    ## set initial positions
    positions_init = atoms.get_positions()
    
    ## get number of water molecules
    num_H = atoms.get_chemical_symbols().count("H")
    num_O = atoms.get_chemical_symbols().count("O")
    if (num_O * 2) != num_H:
        raise ValueError("Number of O atoms is not twice the number of H atoms")
    num_molecules = num_O
    
    ## get mass of water (in the unit of amu)
    if   isotope == "H2O":
        mass_H, mass_O = mass_H_list[0], mass_O_list[0]
    elif isotope == "D2O":
        mass_H, mass_O = mass_H_list[1], mass_O_list[0]
    ## set masses of atoms
    atoms.set_masses([mass_H] * num_H + [mass_O] * num_O)
    
    ## density (in the unit of kg/m^3)
    density = (mass_H * num_H + mass_O * num_O) * units.amu_2_kg / (box_volume * (units.angstrom_2_m**3))
    
    return atoms, box_lengths, positions_init, num_molecules, density

####################################################################################################
if __name__ == "__main__":

    ##============================================================##
    if 0:
        ## create a new ice crystal structure
        
        # stru_path = "structures/raw/ice08_n16.vasp"
        # supercell_size = [1, 1, 1]
        # box_lengths = [6.00, 6.00, 6.06]
        # save_path = "structures/init/ice08_n016.vasp"
        
        # supercell_size = [2, 2, 2]
        # box_lengths = [12.00, 12.00, 12.12]
        # save_path = "structures/init/ice08_n128.vasp"
        
        # stru_path = "structures/raw/ice10_n2.vasp"
        # supercell_size = [2] * 3
        # box_lengths = [5.4] * 3
        # save_path = "structures/init/ice10_n016.vasp"
        
        # stru_path = "structures/raw/ice10_n2.vasp"
        # supercell_size = [4] * 3
        # box_lengths = [10.8] * 3
        # save_path = "structures/init/ice10_n128.vasp"

        stru_path = "structures/relax/ice08c_n016_p300.00.vasp"
        supercell_size = [1, 1, 1]
        box_lengths = [6.00, 6.00, 6.06]
        save_path = "structures/init/ice08_n016.vasp"

        atoms, box_lengths, positions_init, num_molecules, density \
                        = create_ice_crystal(stru_path, supercell_size, box_lengths)

        print("box_lengths (Angstrom):", box_lengths)
        print("num_molecules:", num_molecules)
        print("density (kg/m^3):", density)
        print("masses:", atoms.get_masses())
        
        # ase.io.write(save_path, atoms)
    
    ##============================================================##
    if 0:
        ## create new strucutres for ice10
        
        stru_path = "structures/init/ice10_n016.vasp"
        list_L = np.arange(5.20, 5.82, 0.01)
        supercell_size = [1, 1, 1]
        
        for L in list_L:
            
            L = np.float64(L)
            box_lengths = [L, L, L]
            atoms, box_lengths, positions_init, num_molecules, density \
                            = create_ice_crystal(stru_path, supercell_size, box_lengths)
            print(f"length (A): {L:.2f}    density (kg/m^3) {density:.4f}")
            
            save_path = f"structures/iceX/ice10_n016_L{L:.2f}.vasp"
            ase.io.write(save_path, atoms)
            
    ##============================================================##
    if 0:

        # list_p = np.arange(10, 150, 1)
        # list_p = [250]
        list_p = [150, 200, 250, 300]
        
        for p in list_p:
            print(f"p: {p:.2f}")
            stru_path = f"/home/zq/zqcodeml/waterice/src/structures/relax_l1x128r4.0/tetra_ice08_n016_p{p:.2f}.vasp"
            output_path = f"/home/zq/zqcodeml/waterice/src/structures/relax_l1x128r4.0_init/tetra_ice08_n016_p{p:.2f}.vasp"
            
            atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(stru_path)
            ase.io.write(output_path, atoms,format="vasp")
        
    if 1:
        
        list_p = [150, 200, 250, 300]

        for p in list_p:
            print(f"p: {p:.2f}")
            stru_path = f"/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p{p:.2f}.vasp"
            output_path = f"/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p{p:.2f}_v.vasp"
            
            atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(stru_path)
            atoms = atoms[atoms.get_atomic_numbers().argsort()]
            ase.io.write(output_path, atoms,format="vasp")