#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/Heat_1/DIM_2/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/pytorch_BO/bin/python main_gpu_new.py -cfg config/Heat_1/DIM_2/NeuralBO_heat2.json -gpu_id 0
