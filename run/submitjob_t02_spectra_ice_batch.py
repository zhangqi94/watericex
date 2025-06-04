import os
import sys
import subprocess
import runtools
import time

####################################################################################################
cpus = "6"
nodelist = ""

# partition = "v100"
# gpus = "1"
# num_devices = 1


# partition = "a100"
# gpus = "A100_80G:1"
# num_devices = 1

# partition = "a800"
# gpus = "1"
# num_devices = 1

# partition = "debug"
# gpus = "A100_40D"
# num_devices = 1


partition = "debug"
gpus = "A800D"
num_devices = 1



####################################################################################################
f1 = "/data/zhangqidata/iceX/train_temp0_t02/H2O_ice08_n016_p_40.00_t_1.0_lev_1_flw_[8_256_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_006000.pkl"
f2 = "/data/zhangqidata/iceX/train_temp0_t02/H2O_ice08_n016_p_50.00_t_1.0_lev_1_flw_[8_256_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[1024_1]_key_42/epoch_006000.pkl"
s1 = "H2O_ice08_n016_p_40.00"
s2 = "H2O_ice08_n016_p_50.00"
list_f = [f1, f2]
list_s = [s1, s2]


for i in range(len(list_f)):
    # Define parameters
    jax_mem_frac=0.50
    batch=2048
    acc_steps=8
    num_devices=1
    seed=43

    load_ckpt=list_f[i]

    mace_model_path=f"src/macemodel/mace_iceX_l1x128r4.0.model"
    mace_batch_size=128  ## for a800
    # mace_batch_size=32  ## for v100
    mace_dtype="float32"
    compute_stress=0

    num_spectral_levels=1
    hutchinson=1

    mc_therm=5
    mc_steps=3000
    mc_stddev=0.01

    folder="/data/zhangqidata/iceX/spectra_temp0/"
    input_string=list_s[i]

    ####################################################################################################
    #==== job name ====
    mode_str = (f"{input_string}")
    van_str = f"_lev_{num_spectral_levels}"
    mcmc_str = f"_mc_[{mc_therm}_{mc_steps}_{mc_stddev}]"
    bth_str = f"_bth_[{batch}_{acc_steps}]_key_{seed}"

    job_name = "jobspectra_" + mode_str + van_str + mcmc_str + bth_str

    ####################################################################################################
    #==== python script ====
    pyscript = f"""
cd /home/zhangqi/t02code/waterice
python3 main_spectra.py \\
    --jax_mem_frac {jax_mem_frac} \\
    --batch {batch} \\
    --acc_steps {acc_steps} \\
    --num_devices {num_devices} \\
    --seed {seed} \\
    --load_ckpt "{load_ckpt}" \\
    --mace_model_path "{mace_model_path}" \\
    --mace_batch_size {mace_batch_size} \\
    --mace_dtype {mace_dtype} \\
    --compute_stress {compute_stress} \\
    --num_spectral_levels {num_spectral_levels} \\
    --hutchinson {hutchinson} \\
    --mc_therm {mc_therm} \\
    --mc_steps {mc_steps} \\
    --mc_stddev {mc_stddev} \\
    --folder "{folder}" \\
    --input_string "{input_string}" \\
    """

    ####################################################################################################
    #==== gpu script ====
    gpuscript = f"""
#SBATCH --begin=2025-03-21T02:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}-%j.out
"""

    slurm_script = runtools.generate_slurm_script_singularity(gpuscript, pyscript)
    file_name = job_name + ".sh"
    print(file_name)

    ####################################################################################################
    #==== submit job ====
    runtools.write_slurm_script_to_file(slurm_script, file_name)
    runtools.submit_slurm_script(file_name)

    time.sleep(0.1)
    
"""
cd /home/zhangqi/t02code/waterice/run
python3 submitjob_t02_spectra_ice_batch.py
"""
