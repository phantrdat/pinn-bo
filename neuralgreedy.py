import os
from random import sample

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from base_networks import BaseNN
from scipy.stats import multivariate_normal as mn
from torch.optim.lr_scheduler import ExponentialLR

import math
import time
# from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class NeuralGreedy():
	"""Neural Greedy.
	"""
	def __init__(self, cfg):
		self.dim = cfg.dimension
		self.n_iter = cfg.n_iter

		self.warm_up = 5*self.dim
		if self.warm_up < int(0.025*self.n_iter):
			self.warm_up = int(0.025*self.n_iter)
		if self.warm_up > int(0.075*self.n_iter):
			self.warm_up = int(0.075*self.n_iter)

		self.scale_parameter = cfg.scale_parameter
		self.weight_decay = self.scale_parameter**2
		self.activation = cfg.activation

		
		self.update_cycle = cfg.update_cycle
				
		# hidden size of the NN layers
		self.hidden_size = cfg.W
		# number of layers
		self.n_layers = cfg.L
		# NN parameters
		self.learning_rate = cfg.learning_rate
		self.epochs = cfg.epochs

		self.use_cuda = cfg.use_cuda
		self.device = torch.device(0 if torch.cuda.is_available() and self.use_cuda else 'cpu')

		# neural network
		self.model = BaseNN(input_size=self.dim,
						   hidden_size=self.hidden_size,
						   n_layers=self.n_layers,
						   p=0.0,
						   activation=self.activation).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		
		if cfg.use_lr_scheduler:
			self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
		else:
			self.scheduler = None

		self.iteration = 0
		self.use_local_optimizer = cfg.use_local_optimizer
		self.normalized_inputs = cfg.normalized_inputs
		self.normalized_outputs = cfg.normalized_outputs
		self.acqf_optimizer = cfg.acqf_optimizer
		self.acqf_lr= cfg.acqf_lr
		self.acqf_epochs = cfg.acqf_epochs
		self.n_raw_samples = cfg.n_raw_samples
		self.n_restart = cfg.n_restart


		# Algorithm objective optimization setting
		self.objective_type = cfg.objective_type
		self.n_init = cfg.n_init

		self.X_train = None
		self.Y_train = None
	
	def predict(self, x):
		"""Predict reward.
		"""
		# eval mode
		self.model.eval()
		return self.scale_parameter*self.model.forward(x).detach().squeeze()

	def train(self, x_train, y_train):
		""" Train neural approximator """
		# train mode
		features = x_train.to(self.device).double()
		targets = y_train.to(self.device).double()
		self.model.train()
		loss = torch.tensor(0)
		print(f"** Train NeuralGreedy with {self.epochs} epochs")
		for i in (range(self.epochs)):
			y_pred = self.model.forward(features).squeeze().double()
			y_pred =  self.scale_parameter*y_pred
			loss = nn.MSELoss()(y_pred, targets).double() 			
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if self.scheduler !=None:
				if i%1000==0:
					self.scheduler.step()
		return loss
	
	def Greedy(self, X):
		return self.predict(X)
	
	def optimize_acquisition(self, X, bounds):

		if self.acqf_optimizer == 'Adam':
			adam_optimizer = torch.optim.Adam([X], lr=self.acqf_lr)

			for i in range(self.acqf_epochs):
				adam_optimizer.zero_grad()
				values = self.Greedy(X)
				values.requires_grad = True
				objective = values.sum()
				objective.backward()
				adam_optimizer.step()
				for j, (lb, ub) in enumerate(bounds):
					X.data[...,j].clamp_(lb, ub)
		
		# Add customize acquisition function here

		return X, self.Greedy(X)
	
	def minimize(self, objective):
		torch.manual_seed(0)
		init_points = objective.generate_samples(self.n_init)
		print("Initial points:", init_points)

		self.X_train = init_points['features'].to(self.device)
		self.Y_train = init_points['observations'].to(self.device)

		# Pick the last point of init_points to draw plot
		if self.objective_type == 'synthetic':
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		else:
			optimal_values = [init_points['observations'][-1].item()]

		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)
		X_mean = (objective.max + objective.min)/2
		X_std = torch.abs(objective.max  - X_mean)
		
		if len(self.X_train) != 0 and len(self.Y_train)!=0:
			print("**Fitting known dataset**")
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train
			if self.normalized_outputs:
				print("Training with normalized outputs")
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train
			loss =  self.train(X_train, Y_train)
		
		
		self.model.eval()
		torch.manual_seed(1504)
		x_test = objective.generate_features(10000).to(self.device)
		y_gt = torch.Tensor([objective.value(x, is_noise=False) for x in x_test]).to(self.device)
		if self.normalized_inputs:
			x_test = (x_test-X_mean)/X_std
		
		Y_pred = self.predict(x_test)
		print(f"Eval MSE at step 0", nn.MSELoss()(y_gt, Y_pred.squeeze()).double())
	

		# Phase 1: Explore freely 
		warm_up_points = objective.generate_samples(self.warm_up)
		X_warm_up = warm_up_points['features'].to(self.device)
		Y_warm_up = warm_up_points['observations'].to(self.device)
		
		if self.objective_type == 'synthetic':
			optimal_values += [objective.value(x, is_noise=False).item() for x in X_warm_up]
		else:
			optimal_values += [observation.item() for observation in Y_warm_up]
		
		self.X_train = torch.cat([self.X_train, X_warm_up])
		self.Y_train = torch.cat([self.Y_train, Y_warm_up])
		
		

		if len(self.X_train) != 0 and len(self.Y_train)!=0:
			print("**Fitting explored dataset**")
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train
			if self.normalized_outputs:
				print("Training with normalized outputs")
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train
			self.train(X_train, Y_train)
		
		# Phase 2: 
		for T in range(self.warm_up, self.n_iter):
			print(f"----------NeuralGreedy - Optimization round {T+1}/{self.n_iter}----------")
						
			self.iteration = T+1

			frac, whole = math.modf(time.time())
			try:
				seed = int(whole/(10000*frac)) + T
				torch.manual_seed(seed)
			except:
				torch.manual_seed(1111)
			
			X_cand = objective.generate_features(self.n_raw_samples).to(self.device)
			if self.normalized_inputs:
				X_cand = (X_cand - X_mean)/X_std
			self.model.eval()
			Y_cand = self.Greedy(X_cand)
			
			if self.use_local_optimizer:
				indices = torch.topk(Y_cand, self.n_restart, largest=False).indices
				x_start = X_cand[indices]
				if self.normalized_inputs:
					lb = (objective.min - X_mean)/X_std
					ub = (objective.max - X_mean)/X_std
				else:
					lb = objective.min.to(self.device)
					ub = objective.max.to(self.device)
				x_start.requires_grad = True
				bounds = [(lb[k], ub[k]) for k in range(x_start.shape[1])]
				x_start.requires_grad = True

				x, samples_y = self.optimize_acquisition(X=x_start, bounds= bounds)

				min_idx = torch.argmin(samples_y)
				X_next = x[min_idx]
				acq_value = samples_y[min_idx]
			else:
				min_idx= torch.argmin(Y_cand)
				X_next = X_cand[min_idx]
				acq_value = Y_cand[min_idx]
			
			if self.normalized_inputs:
				X_next = (X_next*X_std + X_mean).detach()
			else:
				X_next = X_next.detach()
			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			
			observation = objective.value(X_next)
			
			if type=='syn':
				true_value = objective.value(X_next, is_noise=False)
			else:
				true_value = observation

			if observation.device.type =='cpu':
				observation = observation.to(self.device)
			if objective.dim >= 2:
				self.Y_train = torch.cat([self.Y_train, observation.unsqueeze(-1)])
			else:
				self.Y_train = torch.cat([self.Y_train, observation])

			if (T+1) % self.update_cycle == 0:
				# self.epochs = 25*(T+1) if T>0 else self.epochs
				if self.normalized_inputs:				
					X_train = (self.X_train - X_mean)/X_std
				else:
					X_train = self.X_train				
				if self.normalized_outputs:
					Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
				else:
					Y_train = self.Y_train				
				
				loss = self.train(X_train, Y_train)
				
				self.model.eval()
				
				
				Y_pred = self.predict(x_test)
				print(f"Eval MSE at step {T+1}", nn.MSELoss()(y_gt, Y_pred.squeeze()).double())
			
			
			print(f"** Iters [{T+1}/{self.n_iter}], current value = {true_value}, acq value: {acq_value}",)
			optimal_values.append(true_value.item())
			
		return optimal_values

	def minimize_discrete(self, objective):
		# ep = self.epochs
		torch.manual_seed(0)
		X_data = torch.DoubleTensor(objective.vectorized_inputs).to(self.device)
		
		init_indexes = torch.randperm(X_data.shape[0])[:objective.dim+1]
		init_features = X_data[init_indexes]


		self.X_train = init_features

		if type(objective.Y_data).__name__ == 'NoneType':
			observation = [objective.value(x) for x in self.X_train]
		else:
			observation = [objective.value(idx) for idx in init_indexes]
		

		self.Y_train = torch.DoubleTensor(observation).to(self.device)

		if type(objective.Y_data).__name__ == "NoneType":
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		else:
			optimal_values = [objective.value(init_indexes[-1])]

		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)
		X_mean = (objective.max + objective.min)/2
		X_std = torch.abs(objective.max  - X_mean)

		if len(self.X_train) != 0 and len(self.Y_train)!=0:

			print("**Fitting known dataset**")
			X_train = (self.X_train - X_mean)/X_std
			if self.normalized_outputs:
				print("Training with normalized outputs")
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train
			self.train(X_train, Y_train)
		

		# Phase 1: Explore freely

		warm_up_points = torch.randperm(X_data.shape[0])[:self.warm_up]
		X_warm_up = X_data[warm_up_points].to(self.device)

		if type(objective.Y_data).__name__ == 'NoneType':
			Y_warm_up = torch.DoubleTensor([objective.value(x) for x in X_warm_up]).to(self.device)
		else:
			Y_warm_up = torch.DoubleTensor([objective.value(idx) for idx in warm_up_points]).to(self.device)
		
		if type == 'syn':
			optimal_values += [objective.value(x, is_noise=False).item() for x in X_warm_up]
		else:
			optimal_values += [observation.item() for observation in Y_warm_up]
		
		self.X_train = torch.cat([self.X_train, X_warm_up])
		self.Y_train = torch.cat([self.Y_train, Y_warm_up])

		print(self.X_train.shape, self.Y_train.shape)
		if len(self.X_train) != 0 and len(self.Y_train)!=0:
			print("**Fitting explored dataset**")
			X_train = (self.X_train - X_mean)/X_std
			if self.normalized_outputs:
				print("Training with normalized outputs")
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train
			self.train(X_train, Y_train)

		for T in range(self.warm_up, self.n_iter):
			print(f"----------NeuralGreedy - Optimization round {T+1}/{self.n_iter}----------")
						
			self.iteration = T+1

			frac, whole = math.modf(time.time())
			try:
				seed = int(whole/(10000*frac)) + T
				torch.manual_seed(seed)
			except:
				torch.manual_seed(1111)
			
			t_idx = torch.randperm(X_data.shape[0])[:512]
			X_data_t = X_data[t_idx]

			# if type(objective.Y_data).__name__ != "NoneType":
			# 	Y_data_t = torch.FloatTensor(objective.Y_data[t_idx]).to(self.device)

			X_next, chosen_idx = self.optimize_acquisition_discrete(X_data_t)
		


			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			
			if type(objective.Y_data).__name__ == "NoneType":
				observation = objective.value(X_next)	
			else:
				observation = objective.value(chosen_idx)
			
			if type=='syn':
				true_value = objective.value(X_next, is_noise=False)
			else:
				true_value = observation

			if observation.device.type =='cpu':
				observation = observation.to(self.device)
			self.Y_train = torch.cat([self.Y_train, observation.unsqueeze(0)])

			self.epochs = T+1 if T>0 else self.epochs
				
			X_train = (self.X_train - X_mean)/X_std				
			if self.normalized_outputs:
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train				
			self.train(X_train, Y_train)

			print(f"** Round [{T+1}/{self.n_iter}], current value = {true_value}")
			optimal_values.append(true_value.item())

		return optimal_values
