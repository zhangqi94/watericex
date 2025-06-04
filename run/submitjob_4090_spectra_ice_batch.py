import os
import sys
import subprocess
import runtools
import time


####################################################################################################



f1 = "/mnt/sg/zqcode/iceX/train_temp100_n128_sg/H2O_ice08_n128_p_30.00_t_100.0_lev_20_flw_[8_384_2]_mc_[3000_0.01]_lr_[0.01_2e-05_0.0002_1e-06_0.99_100]_bth_[128_1]_key_42/epoch_003000.pkl"
s1 = "H2O_ice08_n128_p_30.00"
list_f = [f1]
list_s = [s1]

for i in range(len(list_f)):
    # Define parameters
    jax_mem_frac=0.50
    batch=2048
    acc_steps=8
    num_devices=1
    seed=43

    load_ckpt=list_f[i]

    mace_model_path=f"src/macemodel/mace_iceX_l1x128r4.0.model"
    mace_batch_size=1
    compute_stress=0

    num_spectral_levels=1
    hutchinson=1

    mc_therm=4
    mc_steps=3000
    mc_stddev=0.01

    folder="/home/zq/zqcodeml/waterice/data_test/"
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
    conda activate jax0500-torch251-mace
    cd /home/zq/zqcodeml/waterice
    python3 main_spectra.py \\
    --jax_mem_frac {jax_mem_frac} \\
    --batch {batch} \\
    --acc_steps {acc_steps} \\
    --num_devices {num_devices} \\
    --seed {seed} \\
    --load_ckpt "{load_ckpt}" \\
    --mace_model_path "{mace_model_path}" \\
    --mace_batch_size {mace_batch_size} \\
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
    print("--------------------------------------")
    command = pyscript
    print("Running command: ", command)
    # subprocess.run(command)
    log_file_name = f"{job_name}.log" 
    with open(log_file_name, "w") as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)
    
    print("--------------------------------------")
    
"""
conda activate jax0500-torch251-mace
cd /home/zq/zqcodeml/waterice/run
python3 submitjob_4090_spectra_ice_batch.py
"""
