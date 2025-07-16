

import os
import subprocess
import numpy as np
from runtools import *

################################################################################
# partition = "a800,v100"
# gpus = "1"

partition = "a800"
gpus = "1"

# partition = "v100"
# gpus = "1"


################################################################################

##########################################################################################
def submit_job_256(jobdir, strudir):

    pyscript = f"""
deepsolid --config=/DeepSolid/DeepSolid/config/read_poscar.py:/home/zhangqi/t02codeml/watericex_deepsolid/data_mini_stru/{strudir}/POSCAR,1,ccpvdz \\
    --config.batch_size 4096 \\
    --config.log.save_path /home/zhangqi/t02codeml/watericex_deepsolid/data_mini_results/{jobdir}/ \\
    --config.log.save_frequency 60 \\
    --config.optim.iterations 100000 \\
    --config.mcmc.burn_in 500 \\
    --config.mcmc.init_width 0.8 \\
    --config.mcmc.move_width 0.05 \\
    --config.mcmc.adapt_frequency 100 \\
    --config.network.detnet.hidden_dims '((256, 32), (256, 32), (256, 32), (256, 32))'
    """


    gpuscript = f"""
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --job-name={jobdir}
#SBATCH --output={jobdir}-%j.out
"""

    slurm_script = generate_slurm_script_t02(gpuscript, pyscript)
    file_name = jobdir + ".sh"
    print(file_name)
    
    write_slurm_script_to_file(slurm_script, file_name)
    submit_slurm_script(file_name)


##########################################################################################
def submit_job_128(jobdir, strudir):

    pyscript = f"""
deepsolid --config=/DeepSolid/DeepSolid/config/read_poscar.py:/home/zhangqi/t02codeml/watericex_deepsolid/data_mini_stru/{strudir}/POSCAR,1,ccpvdz \\
    --config.batch_size 4096 \\
    --config.log.save_path /home/zhangqi/t02codeml/watericex_deepsolid/data_mini_results/{jobdir}/ \\
    --config.log.save_frequency 60 \\
    --config.optim.iterations 100000 \\
    --config.mcmc.burn_in 500 \\
    --config.mcmc.init_width 0.8 \\
    --config.mcmc.move_width 0.05 \\
    --config.mcmc.adapt_frequency 100 \\
    --config.network.detnet.hidden_dims '((128, 24), (128, 24), (128, 24), (128, 24))'
    """


    gpuscript = f"""
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --job-name={jobdir}
#SBATCH --output={jobdir}-%j.out
"""

    slurm_script = generate_slurm_script_t02(gpuscript, pyscript)
    file_name = jobdir + ".sh"
    print(file_name)
    
    write_slurm_script_to_file(slurm_script, file_name)
    submit_slurm_script(file_name)


##########################################################################################
def submit_job_192(jobdir, strudir):

    pyscript = f"""
deepsolid --config=/DeepSolid/DeepSolid/config/read_poscar.py:/home/zhangqi/t02codeml/watericex_deepsolid/data_mini_stru/{strudir}/POSCAR,1,ccpvdz \\
    --config.batch_size 4096 \\
    --config.log.save_path /home/zhangqi/t02codeml/watericex_deepsolid/data_mini_results/{jobdir}/ \\
    --config.log.save_frequency 60 \\
    --config.optim.iterations 100000 \\
    --config.mcmc.burn_in 500 \\
    --config.mcmc.init_width 0.8 \\
    --config.mcmc.move_width 0.05 \\
    --config.mcmc.adapt_frequency 100 \\
    --config.network.detnet.hidden_dims '((192, 24), (192, 24), (192, 24), (192, 24))'
    """


    gpuscript = f"""
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --job-name={jobdir}
#SBATCH --output={jobdir}-%j.out
"""

    slurm_script = generate_slurm_script_t02(gpuscript, pyscript)
    file_name = jobdir + ".sh"
    print(file_name)
    
    write_slurm_script_to_file(slurm_script, file_name)
    submit_slurm_script(file_name)



################################################################################


# jobdir = f"p_60.00_a_2.809692_d_0.000000_256"
# strudir = f"p_60.00_a_2.809692_d_0.000000"
# submit_job_256(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.160000_256"
# strudir = f"p_60.00_a_2.809692_d_0.160000"
# submit_job_256(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.000000_128"
# strudir = f"p_60.00_a_2.809692_d_0.000000"
# submit_job_128(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.160000_128"
# strudir = f"p_60.00_a_2.809692_d_0.160000"
# submit_job_128(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.000000_192"
# strudir = f"p_60.00_a_2.809692_d_0.000000"
# submit_job_192(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.160000_192"
# strudir = f"p_60.00_a_2.809692_d_0.160000"
# submit_job_192(jobdir, strudir)

# jobdir = f"p_60.00_a_2.809692_d_0.150000_128"
# strudir = f"p_60.00_a_2.809692_d_0.150000"
# submit_job_128(jobdir, strudir)

jobdir = f"p_60.00_a_2.809692_d_0.150000_256"
strudir = f"p_60.00_a_2.809692_d_0.150000"
submit_job_256(jobdir, strudir)

################################################################################
 
"""
conda activate np2
cd /home/zhangqi/t02codeml/watericex_deepsolid/run/
python3 submitjob_t02-one.py
"""
