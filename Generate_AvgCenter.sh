#!/bin/bash
#SBATCH -o ./outs/generate_fileName_avgCenters.out
#SBATCH --partition=shujiang_rent
#SBATCH -J batteryLifeLLM
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
module load cuda/12.1
source /hpc2hdd/home/rtan474/anaconda3/bin/activate /hpc2hdd/home/rtan474/envs/llmpy311
python generate_fileName_avgCenters.py
