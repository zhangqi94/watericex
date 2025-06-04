import numpy as np
import ase
import ase.io

import pathlib

####################################################################################################


list_p = np.arange(10, 141, 1)
list_p = [30, 40, 70, 80, 120]
list_p = [20, 30, 40, 50, 55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 90, 100, 110, 120, 130]

for i in range(len(list_p)):
    p = list_p[i]
    # stru_file = f"/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p{p:.2f}.vasp"
    
    stru_file = pathlib.Path(__file__).resolve().parent.parent.as_posix() \
        + f"/src/structures/relax/ice08c_n016_p{p:.2f}.vasp"
    atoms = ase.io.read(stru_file)
    box_lengths = np.diag(np.array(atoms.get_cell()))
    L = box_lengths[0]
    
    print(f"pressure: {p:.2f}   L: {L:.8f}")



for i in range(len(list_p)):
    p = list_p[i]

    stru_file = pathlib.Path(__file__).resolve().parent.parent.as_posix() \
        + f"/src/structures/relax/ice08c_n016_p{p:.2f}.vasp"
    atoms = ase.io.read(stru_file)
    box_lengths = np.diag(np.array(atoms.get_cell()))
    L = box_lengths[0]
    
    print(f"{p:.2f}    {L:.8f}    {L**3/16:.8f}")   
