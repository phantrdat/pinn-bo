#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/Styblinski-Tang/DIM_10/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/pytorch_BO/bin/python main_gpu_new.py -cfg config/Styblinski-Tang/DIM_10/NeuralBO_styblinski-tang10.json -gpu_id 0
