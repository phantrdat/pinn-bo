#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/DropWave/DIM_2/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/pytorch_BO/bin/python main_gpu_new.py -cfg config/DropWave/DIM_2/PINN_NeuralBO_dropwave2.json -gpu_id 0
