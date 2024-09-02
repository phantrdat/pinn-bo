#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/Cosine_Mixture/DIM_50/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/torch_bo/bin/python main_gpu_new.py -cfg config/Cosine_Mixture/DIM_50/GPUCB_cosine_mixture50.json -gpu_id 0
