#!/bin/bash
#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/BeamDeflection/DIM_1/slurm_%j.out
module load Anaconda3
source activate pytorch_BO
/home/trongp/.conda/envs/torch_bo/bin/python main_gpu_new.py -cfg config/BeamDeflection/DIM_1/GPUCB_beamdeflection1.json -gpu_id 0
