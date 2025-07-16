import os
import sys
import subprocess

####################################################################################################
def write_slurm_script_to_file(script_content, file_name):
    with open(file_name, 'w') as file:
        file.write(script_content)

def submit_slurm_script(file_name):
    result = subprocess.run(['sbatch', file_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Error in job submission: {result.stderr}")

####################################################################################################
#========== singularity ==========
def generate_slurm_script_t02(gpuscript, pyscript):
    slurm_script = f"""#!/bin/bash
{gpuscript}

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
singularity exec --no-home --nv --bind /home/zhangqi/t02codeml:/home/zhangqi/t02codeml \
    /home/zhangqi/images/cuda11.8-deepsolid.sif bash -c \
"
source /deepsolidenv/bin/activate

nvcc --version
which python3
python3 --version
pip show torch
pip show jax
nvidia-smi

{pyscript}
"

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script

