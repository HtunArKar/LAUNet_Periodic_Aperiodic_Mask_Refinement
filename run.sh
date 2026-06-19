#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 2 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4 		        # Specify number of tasks per node
#SBATCH --gpus=8    		                # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200246                     # Specify project name
#SBATCH -J LAUNET_Ablation              # Specify job name

module purge
module load Mamba
conda activate zonni

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=8   # should be obtained from $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python train.py --clean_path  \
--noisy_path  \
--loss_weights 0.2 0.5 0.1 0.5 \
--save_every 10 \
--seed 42 \
--log_interval 100 \
--config_path  \
--base_output_path  \
--checkpoint_path 