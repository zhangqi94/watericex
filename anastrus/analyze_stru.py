import numpy as np
import ase
import ase.io
from ase import Atoms
from pyxtal import pyxtal

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict

"""
taken from https://code.itp.ac.cn/codes/crystal_gpt/-/blob/main/crystalformer/src/utils.py
"""
####################################################################################################
def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

element_list = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


####################################################################################################
def process_one(crystal, atom_types, wyck_types, n_max, tol=0.01):
    """
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    Process one cif string to get G, L, XYZ, A, W

    Args:
      cif: cif string
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      tol: tolerance for pyxtal

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    # crystal = Structure.from_str(cif, fmt='cif')
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    
    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) # we will need at least one empty site for output of L params

    print (g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    print (ws, aa, ww, natoms) 

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    print ('===================================')

    return g, l, fc, aa, ww 


####################################################################################################
if __name__ == '__main__':
    
    # list_p = np.arange(10, 140, 1)
    
    # for p in list_p:
    #     file_name = f"/home/zq/zqcodeml/waterice/src/structures/relax_l1x128r4.0/tetra_ice08_n016_p{p:.2f}.vasp"
    #     file_name = f"/home/zq/zqcodeml/waterice/src/structures/relax_l1x128r4.0/cubic_ice08_n016_p{p:.2f}.vasp"
    #     atoms = ase.io.read(file_name)

    #     crystal = Structure(
    #         lattice=atoms.cell, 
    #         species=atoms.get_chemical_symbols(),  
    #         coords=atoms.get_scaled_positions(),  
    #         coords_are_cartesian=False,  
    #     )
    #     atom_types = 119
    #     wyck_types = 28
    #     tol = 0.01
    #     n_max = 24
        
    #     process_one(crystal, atom_types, wyck_types, n_max, tol=tol)
        
    # structure.to(fmt="poscar", filename="test.vasp")

    #############
    file_name = f"/home/zq/zqcodeml/waterice/src/structures/iceX/ice10_n016_L5.24.vasp"
    atoms = ase.io.read(file_name)

    crystal = Structure(
        lattice=atoms.cell, 
        species=atoms.get_chemical_symbols(),  
        coords=atoms.get_scaled_positions(),  
        coords_are_cartesian=False,  
    )
    atom_types = 119
    wyck_types = 28
    tol = 0.01
    n_max = 24
    
    process_one(crystal, atom_types, wyck_types, n_max, tol=tol)
