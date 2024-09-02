#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/Michalewics/DIM_30/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/torch_bo/bin/python main_gpu_new.py -cfg config/Michalewics/DIM_30/GPEI_michalewics30.json -gpu_id 0
