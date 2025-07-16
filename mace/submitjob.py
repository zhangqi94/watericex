import subprocess

# 定义提交任务的函数
def submit_slurm_job(partition, 
                     gpu, 
                     x, 
                     r,
                     seed,
                     train_file, 
                     test_file
                     ):
    # 创建 Slurm 脚本内容
    slurm_script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task=6
#SBATCH --job-name=mace_iceX_l1x{x}r{r}
#SBATCH --output=mace_iceX_l1x{x}r{r}-%j.out

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

module purge
singularity exec --no-home --nv --bind /data:/data,/home/zhangqi/t02code:/home/zhangqi/t02code \\
    /home/zhangqi/images/cuda12.6-jax0500-torch240-mace.sif bash -c \\
"
source /jaxtorchmace/bin/activate
nvcc --version
which python3
python3 --version
pip show torch
nvidia-smi
cd /home/zhangqi/t02code/mace-train-iceX/

mace_run_train \\
    --name="mace_iceX_l1x{x}r{r}" \\
    --train_file={train_file} \\
    --valid_fraction=0.05 \\
    --test_file={test_file} \\
    --config_type_weights='{{"Default":1.0}}' \\
    --E0s="average" \\
    --model="MACE" \\
    --hidden_irreps='{x}x0e + {x}x1o' \\
    --correlation=3 \\
    --r_max={r} \\
    --forces_weight=1000 \\
    --energy_weight=10 \\
    --energy_key="TotEnergy" \\
    --forces_key="force" \\
    --scheduler_patience=15 \\
    --eval_interval=2 \\
    --max_num_epochs=1500 \\
    --scheduler_patience=15 \\
    --patience=30 \\
    --ema \\
    --seed={seed} \\
    --restart_latest \\
    --default_dtype="float64" \\
    --device=cuda \\
    --batch_size=8 \\
    --enable_cueq=True
    
"

echo
echo ==== Job finished at `date` ====
"""

    # 构建脚本文件名称
    script_filename = f"job_mace_iceX_l1x{x}r{r}.sh"

    # 将脚本写入文件
    with open(script_filename, 'w') as file:
        file.write(slurm_script)

    # # 提交 Slurm 作业
    subprocess.run(['sbatch', script_filename])
    

####################################################################################################

if __name__ == "__main__":


    # train_file = "data_xyz_0228/watericeX_train.xyz"
    # test_file  = "data_xyz_0228/watericeX_test.xyz"

    train_file = "data_xyz_0310/watericeX_train.xyz"
    test_file  = "data_xyz_0310/watericeX_test.xyz"

    ##############
    # partition = "a800"
    # gpu = "1"
    # x = "64"
    # r = "3.0"
    # seed = "131"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    # partition = "a800"
    # gpu = "1"
    # x = "128"
    # r = "3.0"
    # seed = "132"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    # ##############
    # partition = "a800"
    # gpu = "1"
    # x = "64"
    # r = "4.0"
    # seed = "141"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    partition = "a800"
    gpu = "1"
    x = "96"
    r = "4.0"
    seed = "142"
    submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    # partition = "a800"
    # gpu = "1"
    # x = "128"
    # r = "4.0"
    # seed = "143"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)

    partition = "a800"
    gpu = "1"
    x = "192"
    r = "4.0"
    seed = "144"
    submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)

    # ##############
    # partition = "a800"
    # gpu = "1"
    # x = "64"
    # r = "5.0"
    # seed = "151"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    # partition = "a800"
    # gpu = "1"
    # x = "128"
    # r = "5.0"
    # seed = "152"
    # submit_slurm_job(partition, gpu, x, r, seed, train_file, test_file)
    
    
"""
cd /home/zhangqi/t02code/mace-train-iceX
python3 submitjob.py
"""
    