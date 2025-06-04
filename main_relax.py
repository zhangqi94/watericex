import numpy as np
from mace.calculators import MACECalculator
import ase.io
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

## import units and constants
from src.relax import make_mace_calculator, relax_structure
####################################################################################################
# http://www.patorjk.com/software/taag/#p=display&f=Doom&t=%20iceX%20%20relax
print(r"""
  _         __   __            _            
 (_)        \ \ / /           | |           
  _  ___ ___ \ V /    _ __ ___| | __ ___  __
 | |/ __/ _ \/   \   | '__/ _ \ |/ _` \ \/ /
 | | (_|  __/ /^\ \  | | |  __/ | (_| |>  < 
 |_|\___\___\/   \/  |_|  \___|_|\__,_/_/\_\
                                            
      """, flush=True
      )


import argparse
parser = argparse.ArgumentParser(description= "relax structure for ice crystal")
parser.add_argument("--target_pressure", type=float, default=100, 
                    help="target pressure (GPa)"
                    )
parser.add_argument("--fmax", type=float, default=0.0001, 
                    help="max force (eV/A)"
                    )
parser.add_argument("--steps", type=int, default=5000, 
                    help="max number of steps"
                    )
parser.add_argument("--box_type", type=str, default="tetragonal", 
                    help="type of box (cubic or tetragonal)"
                    )
parser.add_argument("--stru_path", type=str, default="structures/init/ice08_n016.vasp", 
                    help="path to structure file"
                    )
parser.add_argument("--model_path", type=str, default="macemodel/mace_iceX_l1x128r4.0.model", 
                    help="path to model file"
                    )
parser.add_argument("--save_path", type=str, default="structures/relax_iceX_l1x128r4.0/tetra_ice08_n016", 
                    help="path to save relaxed structure file"
                    )
args = parser.parse_args()

print("\n========== Initialize parameters ==========")
target_pressure = args.target_pressure
fmax = args.fmax
steps = args.steps
box_type = args.box_type
print("target pressure (GPa):", target_pressure)
print("max force (eV/A):", fmax)
print("max number of steps:", steps)
print("box type:", box_type)

model_path = args.model_path
stru_path = args.stru_path
save_path = args.save_path
print("mace model file path:", model_path)
print("structure file path:", stru_path)
print("relaxed structure save path:", save_path)

print("\n========== Initialize structure ==========")
from src.crystal import create_ice_crystal
atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(stru_path)
calc = make_mace_calculator(model_path)

print("\n========== Relax structure ==========")
import warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

atoms = relax_structure(atoms, 
                        calc, 
                        target_pressure = target_pressure, 
                        fmax = fmax, 
                        steps = steps, 
                        box_type = box_type,
                        )

save_file_path = save_path + f"_p{target_pressure:.2f}.vasp"
ase.io.write(save_file_path, atoms, format="vasp")
print("save relaxed structure to:", save_file_path)


