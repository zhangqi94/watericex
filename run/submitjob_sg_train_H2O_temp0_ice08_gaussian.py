import os
import sys
import subprocess
import runtools
import time

####################################################################################################
cpus = "6"
nodelist = ""

gpus = "1"
num_devices = 1


####################################################################################################
list_p = [61, 62, 63, 64, 66, 67, 68, 69]
list_p = [20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120]

for i in range(len(list_p)):
    # Define parameters
    jax_mem_frac=0.60
    batch=1024
    num_devices=1
    seed=42
    acc_steps=1

    press = list_p[i]
    init_stru_path=f"src/structures/relax/ice08c_n016_p{press:.2f}.vasp"
    init_box_path="None"
    load_bond_indices="src/structures/bond_indices/ice08c_n016.txt"
    isotope="H2O"
    temperature=1.0
    num_levels=1
    # temperature=50.0
    # num_levels=10
    
    mace_model_path=f"src/macemodel/mace_iceX_l1x128r4.0.model"
    mace_batch_size=32  ## for v100
    mace_dtype="float32"
    compute_stress=1

    flow_layers=0
    flow_width=256
    flow_depth=2
    hutchinson=1

    lr_class=1e-2
    lr_quant=1e-3
    min_lr_class=2e-4
    min_lr_quant=1e-4
    decay_rate=0.99
    decay_steps=100
    decay_begin=1000
    clip_factor=5.0

    mc_therm=10
    mc_steps=3000
    mc_stddev=0.01

    folder="/zqcode/iceX/train_temp0_gaussian_sg/"
    input_string=f"{isotope}_ice08_n016_p_{press:.2f}"
    epoch_finished=0
    epoch_total=6000
    epoch_ckpt=1000
    load_ckpt="None"

    ####################################################################################################
    #==== job name ====
    mode_str = (f"{input_string}" + f"_t_{temperature}")
    van_str = f"_lev_{num_levels}"
    flw_str = f"_flw_[{flow_layers}_{flow_width}_{flow_depth}]"
    mcmc_str = f"_mc_[{mc_steps}_{mc_stddev}]"
    opt_str = f"_lr_[{lr_class}_{lr_quant}_{min_lr_class}_{min_lr_quant}_{decay_rate}_{decay_steps}]"
    bth_str = f"_bth_[{batch}_{acc_steps}]_key_{seed}"
    job_name = "jobtrain_" + mode_str + van_str + flw_str + mcmc_str + opt_str + bth_str

    ####################################################################################################
    #==== python script ====
    pyscript = f"""
cd /zqcode/waterice
python3 main_train.py \\
    --jax_mem_frac {jax_mem_frac} \\
    --batch {batch} \\
    --num_devices {num_devices} \\
    --seed {seed} \\
    --acc_steps {acc_steps} \\
    --init_stru_path "{init_stru_path}" \\
    --init_box_path "{init_box_path}" \\
    --load_bond_indices "{load_bond_indices}" \\
    --isotope {isotope} \\
    --temperature {temperature} \\
    --mace_model_path "{mace_model_path}" \\
    --mace_batch_size {mace_batch_size} \\
    --mace_dtype {mace_dtype} \\
    --compute_stress {compute_stress} \\
    --num_levels {num_levels} \\
    --flow_layers {flow_layers} \\
    --flow_width {flow_width} \\
    --flow_depth {flow_depth} \\
    --hutchinson {hutchinson} \\
    --lr_class {lr_class} \\
    --lr_quant {lr_quant} \\
    --min_lr_class {min_lr_class} \\
    --min_lr_quant {min_lr_quant} \\
    --decay_rate {decay_rate} \\
    --decay_steps {decay_steps} \\
    --decay_begin {decay_begin} \\
    --clip_factor {clip_factor} \\
    --mc_therm {mc_therm} \\
    --mc_steps {mc_steps} \\
    --mc_stddev {mc_stddev} \\
    --folder "{folder}" \\
    --input_string "{input_string}" \\
    --epoch_finished {epoch_finished} \\
    --epoch_total {epoch_total} \\
    --epoch_ckpt {epoch_ckpt} \\
    --load_ckpt "{load_ckpt}"
    """

    ####################################################################################################
    #==== gpu script ====
    gpuscript = f"""
#SBATCH -t 99:00:00
#SBATCH --partition=dzagnormal
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}-%j.out
"""

    slurm_script = runtools.generate_slurm_script_singularity_sg(gpuscript, pyscript)
    file_name = job_name + ".sh"
    print(file_name)

    ####################################################################################################
    #==== submit job ====
    runtools.write_slurm_script_to_file(slurm_script, file_name)
    runtools.submit_slurm_script(file_name)

    time.sleep(0.1)
    
"""
cd /work/home/ackl68vqh5/zqcode/waterice/run
python3 submitjob_sg_train_H2O_temp0_ice08_gaussian_batch.py
"""
