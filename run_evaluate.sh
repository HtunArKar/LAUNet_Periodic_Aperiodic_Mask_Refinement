#!/bin/bash -l
#SBATCH -p gpu                     #specify partition
#SBATCH -N 1 -c 16                 #specify number of nodes
#SBATCH --gpus-per-task=1          #specify number of gpu per task
#SBATCH --ntasks-per-node=4        #specify tasks per node
#SBATCH -t 24:00:00                #job time limit <hr:min:sec>
#SBATCH -J Evaluation              #job name
#SBATCH -A lt200246                #specify your account ID

module load Mamba
conda activate zonni


python evaluate.py --clean_path  \
--noisy_path  \
--log_interval 100 \
--config_path  \
--se_model_path  \
--asr_model_path /lustrefs/disk/project/lt200246-mmacma/ZonNi/Voice_Denoising_Modified/T_Whisper/ \
--gt_path 