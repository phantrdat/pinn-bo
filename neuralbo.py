import os
from random import sample
from tkinter.messagebox import NO

from objectives import Objective
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
from base_networks import BaseNN
from scipy.stats import multivariate_normal as mn
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ExponentialLR

import math
import copy
import time


class NeuralBO():
	"""Thompson Sampling using Deep Neural Networks.
	"""
	def __init__(self, cfg
				 ):

		# L2 regularization strength
		self.dim = cfg.dimension
		self.n_iter = cfg.n_iter
		self.activation = cfg.activation
		self.weight_decay = cfg.weight_decay

		self.update_cycle = cfg.update_cycle
		
		# hidden size of the NN layers
		self.hidden_size = cfg.W
		# number of layers
		self.n_layers = cfg.L

		# NN hyper-parameters
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
		
		self.init_model = copy.deepcopy(self.model)
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

		# Algorithm specific configs
		self.use_matrix_inversion_appoximation = cfg.use_matrix_inversion_appoximation
		if self.use_matrix_inversion_appoximation:
			self.U = (torch.eye(self.approximator_dim)*self.weight_decay).double().to(self.device)

		else:
			self.U_inv = (torch.eye(self.approximator_dim)/self.weight_decay).double().to(self.device)
		
		self.exploration_coeff = cfg.exploration_coeff
		
		self.X_train = None
		self.Y_train = None
		self.feature_mode = cfg.feature_mode
		# self.run_idx = run_idx
	@property
	def approximator_dim(self):
		"""Sum of the dimensions of all trainable layers in the network.
		"""
		return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

	
	def calculate_gradient(self, x):
		"""Get gradient of network prediction w.r.t network weights.
		"""
		# x = torch.FloatTensor(x).to(self.device)
		grads_approx = None
		if self.feature_mode=='static':
			y = self.init_model(x)
			grads_approx = [None]*y.shape[0]
			for i in range(y.shape[0]):
				self.init_model.zero_grad()
				y[i].backward(retain_graph=True)

				grads_approx[i] = torch.cat(
						[w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.init_model.parameters() if w.requires_grad]
				).to(self.device)
		elif self.feature_mode =='dynamic':
			y = self.model(x)
			grads_approx = [None]*y.shape[0]
			for i in range(y.shape[0]):
				self.model.zero_grad()
				y[i].backward(retain_graph=True)

				grads_approx[i] = torch.cat(
						[w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
				).to(self.device)

		grads_approx = torch.stack(grads_approx)
		return grads_approx.T


	def predict(self, X):
		"""Predict reward.
		"""
		# eval mode
		self.model.eval()
		return self.model.forward(X).detach().squeeze()

	def sample_reward(self, x):
		
		grads = self.calculate_gradient(x)

		if self.use_matrix_inversion_appoximation:
		# grad_approx = self.calculate_gradient(x)
			U_inv = 1/torch.diag(self.U,0)
			U_inv = torch.diag(U_inv)
			AinvGT = torch.matmul(grads.T, U_inv)
			GAinvGT = torch.matmul(AinvGT, grads)
			variances = self.weight_decay*torch.diag(GAinvGT)
		else:
			AinvGT = torch.matmul(grads.T, self.U_inv)
			GAinvGT = torch.matmul(AinvGT, grads)
			variances = self.weight_decay*torch.diag(GAinvGT)
	

		explored_std = self.exploration_coeff*torch.sqrt(variances)
		batch_mu_hat = self.predict(x)


		r_hat = torch.Tensor([Normal(m,v).sample() if m!=0 and v>0 else torch.inf for m,v in zip(batch_mu_hat, explored_std)])
		r_hat.requires_grad=True
		return r_hat, (batch_mu_hat, explored_std)

	def train(self, x_train, y_train):
		"""Train neural approximator."""
		# train mode
		x_train = x_train.double()
		y_train = y_train.double()
		self.model.train()
		loss = torch.tensor(0.0)
		for i in (range(self.epochs)):
			y_pred = self.model.forward(x_train).squeeze().double()
			loss = nn.MSELoss()(y_pred, y_train).double()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if self.scheduler !=None:
				if i%1000==0:
					self.scheduler.step()
			# if i%10000==0:
			# 	print(f"NeuralBO epoch {i}, loss {loss.item()}")
		return loss

			
	
	def update_A_inv(self, x_t):

		# Implement matrix inversion with Sherman-Morrison method # 
	
		def inv_sherman_morrison(u, A_inv):
			"""Inverse of a matrix with rank 1 update.
			"""
			Au = torch.matmul(A_inv, u).squeeze(-1)
			A_inv -= torch.outer(Au, Au)/(1+torch.matmul(u.T, Au))
			return A_inv
	
		grad_approx = self.calculate_gradient(x_t)
		if self.use_matrix_inversion_appoximation ==True:
			self.U = self.U + torch.matmul(grad_approx, grad_approx.T)
		else:
			self.U_inv = inv_sherman_morrison(
				grad_approx,
				self.U_inv
			)
		
	def TS(self, X):
		return self.sample_reward(X)	
	
	def optimize_acquisition(self, X, bounds):
		if self.acqf_optimizer == 'Adam':
			adam_optimizer = torch.optim.Adam([X], lr=self.acqf_lr)

			for i in range(self.acqf_epochs):
				adam_optimizer.zero_grad()
				values, _ = self.TS(X)
				objective = values.sum()
				objective.backward()
				adam_optimizer.step()
				for j, (lb, ub) in enumerate(bounds):
					X.data[...,j].clamp_(lb, ub)		
		samples, (predictive_means, predictive_stds) = self.TS(X)
		return X, samples, (predictive_means, predictive_stds)
	def optimize_acquisition_discrete(self, X):
		batch_size = 5
		
		chunks = torch.split(X, batch_size)
		acqf_values = []
		for b in chunks:
			acqf_v, (pmean, pvar) =  self.TS(b)
			acqf_values += acqf_v
		acqf_values = torch.stack(acqf_values)
		return X[torch.argmin(acqf_values)], torch.argmin(acqf_values)


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
			loss = self.train(X_train, Y_train)
		
		self.model.eval()
		torch.manual_seed(1504)
		x_test = objective.generate_features(10000).to(self.device)
			
		y_gt = torch.Tensor([objective.value(x, is_noise=False) for x in x_test]).to(self.device)
		if self.normalized_inputs:
			x_test = (x_test- X_mean)/X_std
		
		Y_pred = self.model(x_test)
		print(f"Eval MSE at step 0", nn.MSELoss()(y_gt, Y_pred.squeeze()).double())

		for T in range(self.n_iter):
			print(f"----------NeuralBO - Optimization round {T+1}/{self.n_iter}----------")
			self.iteration = T+1
			if self.exploration_coeff == 0.0:
				multiplier = (self.iteration)/np.log(self.iteration+1)
				multiplier = multiplier*np.log(1+self.iteration*multiplier)
				self.exploration_coeff = np.sqrt(multiplier)
			
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
			Y_cand, (pmeans, pstds) = self.TS(X_cand)

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
				x, samples_x, (pmeans, pstds) = self.optimize_acquisition(X=x_start, bounds=bounds)
				min_idx = torch.argmin(samples_x)
				predictive_mean, predictive_std = pmeans[min_idx], pstds[min_idx]
				X_next = x[min_idx]
			else:
				min_idx= torch.argmin(Y_cand)
				predictive_mean, predictive_std = pmeans[min_idx], pstds[min_idx]
				X_next = X_cand[min_idx]

			self.update_A_inv(X_next)

			if self.normalized_inputs:
				X_next = (X_next*X_std + X_mean).detach()
			else:
				X_next = X_next.detach()

			
			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			

			observation = objective.value(X_next)
			
			if self.objective_type == 'synthetic':
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
				print(f"** Train NeuralBO with {self.epochs} epochs")
				
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
				Y_pred = self.model(x_test)
				print(f"Eval MSE at step {T+1}", nn.MSELoss()(y_gt, Y_pred.squeeze()).double())
			print(f"** Round [{T+1}/{self.n_iter}], current value = {true_value}, predictive mean {predictive_mean}, predictive std {predictive_std}")
			optimal_values.append(true_value.item())

			

		return optimal_values
	def minimize_discrete(self, objective):
		# ep = self.epochs
		
		torch.manual_seed(0)
		X_data = torch.FloatTensor(objective.vectorized_inputs).to(self.device)
		
		
		init_indexes = torch.randperm(X_data.shape[0])[:objective.dim+1]
		init_features = X_data[init_indexes]


		self.X_train = init_features

		if type(objective.Y_data).__name__ == 'NoneType':
			observation = [objective.value(x) for x in self.X_train]
		else:
			# observation = objective.Y_data[init_indexes]
			observation = [objective.value(idx) for idx in init_indexes]
		
		
		self.Y_train = torch.FloatTensor(observation).to(self.device)

		if type(objective.Y_data).__name__ == "NoneType":
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		else:
			# optimal_values = [objective.Y_data[init_indexes[-1]]]
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
			self.train(X_train, Y_train, train_batch_size=self.train_batch_size)
		for T in range(self.n_iter):
			self.iteration = T+1
			
			if self.exploration_variance == 0.0:
				multiplier = (self.iteration)/np.log(self.iteration+1)
				multiplier = multiplier*np.log(1+self.iteration*multiplier)
				self.exploration_coeff = np.sqrt(multiplier)
			else:
				self.exploration_coeff = self.exploration_variance


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
			self.update_A_inv(X_next)

			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			
			if type(objective.Y_data).__name__ == "NoneType":
				observation = objective.value(X_next)
				true_value = objective.value(X_next, is_noise=False)
				
			else:
				observation = objective.value(t_idx[chosen_idx])
				true_value = observation
			
			
			if observation.device.type =='cpu':
				observation = observation.to(self.device)

				

			self.Y_train = torch.cat([self.Y_train, observation.unsqueeze(0)])

			if (T+1) % self.update_cycle == 0:
				# self.epochs = 25*(T+1)

				print(f"** Train NeuralTS with {self.epochs} epochs")

				X_train = (self.X_train - X_mean)/X_std
				if self.normalized_outputs:
					Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
				else:
					Y_train = self.Y_train
				self.train(X_train, Y_train, train_batch_size=self.train_batch_size)
			print(f"** Round [{T+1}/{self.n_iter}], current value = {true_value}")
			optimal_values.append(true_value.item())
		return optimal_values




	