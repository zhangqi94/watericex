import os
import subprocess

# Activate the desired conda environment
# conda_env = "jax0500-torch240-mace"
# os.system(f"conda activate {conda_env}")

# Change to the target directory
target_dir = "/home/zq/zqcode/waterice"
os.chdir(target_dir)

# Define the pressure range
pressure_start = 10
pressure_end = 140
pressure_step = 1

# pressure_start = 150
# pressure_end = 300
# pressure_step = 50

for pressure in range(pressure_start, pressure_end + 1, pressure_step):
    print("--------------------------------------")
    print(f"Running with pressure: {pressure}")

    command = [
        "python3", "main_relax.py",
        "--target_pressure", str(pressure),
        "--box_type", "cubic",
        "--model_path", "src/macemodel/mace_iceX_l1x128r4.0.model",
        "--stru_path", "src/structures/init/ice08_n016.vasp",
        "--save_path", "src/structures/relax_l1x128r4.0/ice08_cubic_n016",
    ]
    
    
    print("Running command: ", command)
    
    # Run the command
    subprocess.run(command)
    
    print("--------------------------------------")

"""
conda activate jax0500-torch251-mace
cd /home/zq/zqcodeml/waterice/run
python3 submitjob_zqpc_relax.py
"""