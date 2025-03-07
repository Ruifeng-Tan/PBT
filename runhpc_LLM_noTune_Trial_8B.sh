#!/bin/bash
#SBATCH -o ./outs/8B_MIX_large_v12_SDCM.%j.out
#SBATCH --partition=shujiang_rent
#SBATCH -J batteryLifeLLM
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
module load cuda/12.1
source /hpc2hdd/home/rtan474/anaconda3/bin/activate /hpc2hdd/home/rtan474/envs/llmpy311
sh scripts/Llama31I_8B.sh
