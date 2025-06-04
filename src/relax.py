import numpy as np
from mace.calculators import MACECalculator
import ase.io
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

## import units and constants
try:
    from src.units import units
    from src.potentialmace import make_mace_calculator
except:
    from units import units
    from potentialmace import make_mace_calculator

####################################################################################################
def relax_structure(atoms,
                    calc,
                    target_pressure = 10.0,
                    fmax = 0.0001,
                    steps = 5000,
                    box_type = "tetragonal",
                    print_info = True,
                    ):
    
    atoms.calc = calc

    if print_info:
        box_lengths = np.diag(np.array(atoms.get_cell()))
        stress = atoms.get_stress()
        pressure = -np.mean(stress[:3])
        print("initial box lengths (angstrom):", box_lengths)
        print("initial stress (GPa):", stress * units.eVangstrom_2_GPa)
        print("initial pressure (GPa):", pressure * units.eVangstrom_2_GPa)

    ## start relaxation
    fcf = FrechetCellFilter(atoms, 
                            scalar_pressure = target_pressure / units.eVangstrom_2_GPa,
                            )
    FIRE(fcf).run(fmax=fmax, steps=steps)

    ## adjust box shape
    box_lengths = np.diag(np.array(atoms.get_cell()))

    if   box_type == "cubic":
        average_value = box_lengths.mean()
        box_lengths = np.full_like(box_lengths, average_value)
        atoms.set_cell(box_lengths, scale_atoms = True,) 
        
    elif box_type == "tetragonal":
        average_of_first_two = box_lengths[:2].mean()
        box_lengths = np.array([average_of_first_two, average_of_first_two, box_lengths[2]])
        atoms.set_cell(box_lengths, scale_atoms = True,) 

    ## get relaxed structure and forces
    if print_info:
        # box_lengths = np.diag(np.array(atoms.get_cell()))
        stress = atoms.get_stress()
        pressure = -np.mean(stress[:3])
        print("final box lengths (angstrom):", box_lengths)
        print("final stress (GPa):", stress * units.eVangstrom_2_GPa)
        print("final pressure (GPa):", pressure * units.eVangstrom_2_GPa)
    
    return atoms


####################################################################################################
if __name__ == "__main__":

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
    parser.add_argument("--save_path", type=str, default="structures/relax/tetra_ice08_n016", 
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
    from crystal import create_ice_crystal
    atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(stru_path)
    calc = make_mace_calculator(model_path)

    print("\n========== Relax structure ==========")
    import warnings
    warnings.filterwarnings("ignore", message="logm result may be inaccurate")
    
    atoms = relax_structure(atoms, calc, 
                            target_pressure = target_pressure, 
                            fmax = fmax, 
                            steps = steps, 
                            box_type = box_type,
                            )

    atoms = atoms[atoms.get_atomic_numbers().argsort()]
    save_file_path = save_path + f"_p{target_pressure:.2f}.vasp"
    ase.io.write(save_file_path, atoms, format="vasp")
    print("save relaxed structure to:", save_file_path)


####################################################################################################







"""
conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice/src
for PRESS in $(seq 10 5 145); do
    echo "--------------------------------------"
    echo "Running with pressure: $PRESS"
    python3 relax.py  --target_pressure $PRESS  --box_type "tetragonal" \
        --stru_path "structures/init/ice08_n016.vasp" \
        --save_path "structures/relax/tetra_ice08_n016"
    echo "--------------------------------------"
done


conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice/src
for PRESS in $(seq 10 5 145); do
    echo "--------------------------------------"
    echo "Running with pressure: $PRESS"
    python3 relax.py  --target_pressure $PRESS  --box_type "cubic" \
        --stru_path "structures/init/ice08_n016.vasp" \
        --save_path "structures/relax/cubic_ice08_n016"
    echo "--------------------------------------"
done


conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice/src
for PRESS in $(seq 10 5 145); do
    echo "--------------------------------------"
    echo "Running with pressure: $PRESS"
    python3 relax.py  --target_pressure $PRESS  --box_type "tetragonal" \
        --stru_path "structures/init/ice08_n128.vasp" \
        --save_path "structures/relax/tetra_ice08_n128"
    echo "--------------------------------------"
done


conda activate jax0500-torch240-mace
cd /home/zq/zqcodeml/waterice/src
for PRESS in $(seq 10 5 145); do
    echo "--------------------------------------"
    echo "Running with pressure: $PRESS"
    python3 relax.py  --target_pressure $PRESS  --box_type "cubic" \
        --stru_path "structures/init/ice08_n128.vasp" \
        --save_path "structures/relax/cubic_ice08_n128"
    echo "--------------------------------------"
done
"""