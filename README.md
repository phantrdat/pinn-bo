# Project Name

This repository contains the source code and configurations for our PINN-BO paper submitted to AISTATS 2024. 

## Requirements

Make sure you have the following dependencies installed:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [BoTorch](https://botorch.org/)

You can install them using `pip`:

```bash
pip install torch numpy botorch

To run the experiment with a single GPU, use the following command:
python main_gpu_new -cfg <path_to_config> -gpu_id <id>

To run the experiment with slurm, use the following command:
sbatch run_scripts/<script_to_run>

To plot the results of a single objective function, use file plot1.py
To plot the results of a multiple objective functions, use file plot2.py