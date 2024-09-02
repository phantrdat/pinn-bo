from locale import normalize
from math import inf
import numpy as np
import pickle as pkl
from matplotlib  import pyplot as plt
import os
import torch
import itertools
from objectives import *
import argparse
import json
import numpy as np
import time
from neuralbo import NeuralBO
from pinn_neuralbo import PINN_NeuralBO
from gptorch import GPTorch
from neuralgreedy import NeuralGreedy
from types import SimpleNamespace
import warnings

def fxn():
	warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	fxn()

DATA_DIR = 'data/'
RES_DIR = 'results'
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', type=str, default='config/BeamDeflection/DIM_1/PINN_NeuralBO_beamdeflection1.json', help='Config File')
parser.add_argument('-gpu_id', default=0, help='GPU ID')

functions = {"Michalewics": Michalewics, 
			 "Styblinski-Tang": Styblinski_Tang, 
			 "Rastrigin": Rastrigin,
			 "DropWave": DropWave, 
			 "Cosine_Mixture": Cosine_Mixture, 
			 'Heat_1': Heat1, 
			 'Heat_2': Heat2, 
			 'Heat_3': Heat3,
			 'BeamDeflection':BeamDeflection}

args = parser.parse_args()

GPU_ID = int(args.gpu_id)


def run_opt(alg, objective):
	t1 = time.time()
	optimal_values = alg.minimize(objective)
	t2 = time.time() - t1
	info = {'function_name': objective.func_name, 
			"optimal_values": optimal_values,  
			"dim": objective.dim, 
			"X_train": alg.X_train.cpu(),
			"Y_train": alg.Y_train.cpu(),
			"Running time": t2}
	return info


if __name__ == '__main__':


	configs = json.load(open(args.cfg, "r"))
	print(configs)
	
	configs = SimpleNamespace(**configs)
		
	
	objective = None
	if configs.objective_type =='synthetic':
		objective = functions[configs.function_name](dim=configs.dimension)

	
	if "gp" in configs.algorithm_type.lower():
		
		for run_idx in range(configs.first_run, configs.last_run):
			print("Run:", run_idx)
			alg_obj = GPTorch(cfg=configs)
			
			
			info = run_opt(alg_obj, objective)
			save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ROUNDS_{configs.n_iter}/{configs.algorithm_type}"
			if os.path.isdir(save_root) ==False:
				os.makedirs(save_root)
			
			file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_kernel_{configs.kernel_name}_dim_{info['dim']}.{run_idx}.pkl"
			pkl.dump(info, open(file_name,'wb'))

	if 'neuralbo' == configs.algorithm_type.lower():		
		print("Normalized outputs:", configs.normalized_outputs)
		print("Normalized inputs:", configs.normalized_inputs)
		print("Use matrix inversion approximation:", configs.use_matrix_inversion_appoximation)
		for run_idx in range(configs.first_run, configs.last_run):
			print("Run:", run_idx)
			
			neuralbo = NeuralBO(cfg=configs)
			info = run_opt(neuralbo, objective)

			save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ROUNDS_{configs.n_iter}/{configs.algorithm_type}"
			if os.path.isdir(save_root) ==False:
				os.makedirs(save_root)
			
			file_name = f"{save_root}/{configs.algorithm_type}_L_{configs.L}_w_{configs.W}_{info['function_name']}_dim_{info['dim']}_mode_{configs.feature_mode}_matapprox_{str(configs.use_matrix_inversion_appoximation)}.{run_idx}.pkl"
			pkl.dump(info, open(file_name,'wb'))

	if 'neuralgreedy' == configs.algorithm_type.lower():

		print("Normalized outputs:", configs.normalized_outputs)
		print("Normalized inputs:", configs.normalized_inputs)
		for run_idx in range(configs.first_run, configs.last_run):
			print("Run:", run_idx)

			neural_greedy = NeuralGreedy(cfg=configs)



			info = run_opt(neural_greedy, objective)

			save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ROUNDS_{configs.n_iter}/{configs.algorithm_type}"
			if os.path.isdir(save_root) ==False:
				os.makedirs(save_root)
			
			file_name = f"{save_root}/NeuralGreedy_L_{configs.L}_w_{configs.W}_{info['function_name']}_dim_{info['dim']}.{run_idx}.pkl"
			pkl.dump(info, open(file_name,'wb'))

	if 'pinn_neuralbo' == configs.algorithm_type.lower():
		print("Normalized outputs:", configs.normalized_outputs)
		print("Normalized inputs:", configs.normalized_inputs)
		# for lr in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
		# 	configs.learning_rate = lr
		for run_idx in range(configs.first_run, configs.last_run):
			print(configs.learning_rate)
			print("---------------RUN:", run_idx)
			torch.manual_seed(15)
			pinn_neuralbo = PINN_NeuralBO(obj=objective, cfg=configs)
			info = run_opt(pinn_neuralbo, objective)
			save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ROUNDS_{configs.n_iter}/{configs.algorithm_type}/{configs.algorithm_type}_L_{configs.L}_w_{configs.W}_lr_{configs.learning_rate}_{info['function_name']}_dim_{info['dim']}"
			if os.path.isdir(save_root) ==False:
				os.makedirs(save_root)
			file_name = f"{save_root}/{configs.algorithm_type}_L_{configs.L}_w_{configs.W}_lr_{configs.learning_rate}_{info['function_name']}_dim_{info['dim']}.{run_idx}.pkl"
			pkl.dump(info, open(file_name,'wb'))
		
	


	




		

	
