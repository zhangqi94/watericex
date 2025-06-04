import ase
import ase.io
from ase import Atoms
import numpy as np

####################################################################################################

def create_ice(a=2.8, d=0.1):
    """
    Create an ice structure with given lattice constant and displacement.

    Parameters:
        a (float): The lattice constant of the ice structure.
        d (float): Displacement distance to introduce asymmetry.

    Returns:
        ase.Atoms: The generated ice structure.
    """
    lattice_constant = np.float64(a)
    displacement     = np.float64(d)
    dx = displacement / (np.sqrt(3) * lattice_constant)

    def generate_positions(dx):
        """Generate atomic positions for a single unit cell."""
        return np.array([
            [0, 0, 0],
            [0.50, 0.50, 0.50],
            [0.25 - dx, 0.25 - dx, 0.25 - dx],
            [0.75 + dx, 0.75 + dx, 0.25 - dx],
            [0.75 - dx, 0.25 + dx, 0.75 - dx],
            [0.25 + dx, 0.75 - dx, 0.75 - dx],
        ], dtype=np.float64)

    positions_up = generate_positions( dx)
    positions_dn = generate_positions(-dx)

    # Generate coordinates by applying translations
    translations = [
        [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]
    ]
    positions = np.concatenate([
        positions_up + t if i < 4 else positions_dn + t
        for i, t in enumerate(translations)
    ], axis=0)

    # Define the lattice
    box_lengths = np.array([lattice_constant] * 3) * 2
    atom_positions = positions * lattice_constant
    atom_types = "O2H4" * 8

    # Create ASE atoms object
    atoms = ase.Atoms(
        atom_types,
        positions=atom_positions,
        cell=np.diag(box_lengths),
        pbc=True,
    )

    # Sort atoms by atomic number
    atoms = atoms[atoms.get_atomic_numbers().argsort()]
    
    return atoms

####################################################################################################
if __name__ == '__main__':
    
    ################
    # atoms = create_ice(a=2.8, d=0.1)
    # ase.io.write("test.vasp", atoms)

    ################
    import os
    
    ## around 30GPa, 40GPa, 70GPa, 80GPa, 120GPa
    #np.array([5.818662, 5.710456, 5.489954, 5.434057, 5.248239])/2
    a = 5.818662 / 2
    a = 5.489954 / 2
    a = 5.710456 / 2
    a = 5.434056 / 2
    a = 5.248238 / 2
    
    list_d = np.linspace(0, 0.55, 111)
    
    for i in range(len(list_d)):
        d = list_d[i]
        atoms = create_ice(a, d)
        
        save_path = f"data_dft/a_{a:.6f}_d_{d:.6f}"
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.join(save_path, "POSCAR")
        ase.io.write(file_name, atoms, format="vasp")
    
