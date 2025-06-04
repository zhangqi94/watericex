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

    positions = generate_positions( dx)

    # Define the lattice
    box_lengths = np.array([lattice_constant] * 3)
    atom_positions = positions * lattice_constant
    atom_types = "O2H4"

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
    a = 2.8
    d = 0.3
    
    atoms = create_ice(a, d)
    
    save_path = f"data_fig/"
    os.makedirs(save_path, exist_ok=True)
    file_name = save_path + f"a_{a:.6f}_d_{d:.6f}.vasp"
    ase.io.write(file_name, atoms, format="vasp")
    
