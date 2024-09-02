import os
from random import sample
import torch.autograd as autograd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
from base_networks import BaseNN
import math
import time
from botorch.utils.sampling import batched_multinomial
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from torch.optim.lr_scheduler import ExponentialLR
class PINN_NeuralBO():
	"""PINN using Deep Neural Networks.
	"""
	def __init__(self, obj, cfg):

		# L2 regularization strength
		self.objective = obj
		self.dim = cfg.dimension
		self.nr = cfg.n_residuals
		self.n_iter = cfg.n_iter
		self.activation = cfg.activation
		self.weight_decay = cfg.weight_decay
		self.exploration_coeff = cfg.exploration_coeff 
		self.update_cycle = cfg.update_cycle
				
		# hidden size of the NN layers
		self.hidden_size = cfg.W
		# number of layers
		self.n_layers = cfg.L

		
		# NN hyper-parameters
		self.learning_rate = cfg.learning_rate
		self.epochs = cfg.epochs
		# dropout rate

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

	@property
	def approximator_dim(self):
		"""Sum of the dimensions of all trainable layers in the network.
		"""
		return sum(w.numel() for w in self.model.parameters() if w.requires_grad)


	def predict(self, X):
		""" Predict reward """
		self.model.eval()
		return self.exploration_coeff*self.model.forward(X).detach().squeeze()

	def train(self, x_train, y_train, x_PDE, epochs=10000):
		def compute_laplacian(f, input_tensor):
			"""
			Compute the Laplacian of a function f with respect to the input tensor.
			
			Args:
				f (callable): The function for which the Laplacian is calculated.
				input_tensor (torch.Tensor): Input tensor with shape (n, d).
				
			Returns:
				torch.Tensor: Laplacian with shape (n,).
			"""
			n, d = input_tensor.size()
			input_tensor.requires_grad_(True)

			# Compute the gradient of f with respect to the input tensor
			gradient = torch.autograd.grad(f(input_tensor).sum(), input_tensor, create_graph=True)[0]

			# Compute the Laplacian (sum of second partial derivatives)
			laplacian = torch.zeros(n, dtype=input_tensor.dtype, device=input_tensor.device)
			for i in range(d):
				laplacian += torch.autograd.grad(gradient[:, i].sum(), input_tensor, create_graph=True)[0][:, i]

			return laplacian
				
		def pinn_loss():
			# physics informed loss
			x_pde = x_PDE.to(self.device).double()
			x_pde.requires_grad = True
			y_pred = self.model.forward(x_pde).squeeze()
			if self.objective.func_name =='DropWave':
				grads = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred).to(self.device), retain_graph=True, create_graph=True)[0]
				pinn_l = self.exploration_coeff*(torch.mul(x_pde[:,1], grads[:,0]) - torch.mul(x_pde[:,0], grads[:,1]))
			
			if self.objective.func_name == 'Rastrigin':
				grads = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred).to(self.device), retain_graph=True, create_graph=True)[0]
				lhs = 0.5*torch.sum(x_pde*grads, dim=1) - y_pred
				rhs = 10*torch.sum(torch.cos(2*torch.pi*x_pde) + torch.pi*x_pde*torch.sin(2*torch.pi*x_pde), dim=1) - 10*x_pde.size(1)
				pinn_l = self.exploration_coeff*lhs - rhs
			
			if self.objective.func_name == 'Cosine_Mixture':
				grads = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]
				# pinn_l = torch.sum(grads, dim=1) - torch.sum(2*x_pde - 0.5*torch.pi*torch.sin(5*torch.pi*x_pde), dim=1)
				pinn_l = self.exploration_coeff*grads - (2*x_pde - 0.5*torch.pi*torch.sin(5*torch.pi*x_pde))

			if self.objective.func_name == "Styblinski-Tang":
				grads = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred).to(self.device), retain_graph=True, create_graph=True)[0]
				lhs = torch.sum(grads, dim=1)
				rhs = torch.sum(2*x_pde**3 - 16*x_pde + 2.5, dim=1)
				pinn_l = self.exploration_coeff*lhs - rhs
			if self.objective.func_name == 'Michalewics':
				grads = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred).to(self.device), retain_graph=True, create_graph=True)[0]
				m = 10
				idx = torch.arange(1, x_pde.size(1)+1).float().to(self.device)
				u = 1/(1/torch.tan(x_pde) + (4*m*idx*x_pde/torch.pi)/torch.tan(idx*x_pde**2/torch.pi))
				rhs = torch.sum(u*grads, dim=1)
				pinn_l = y_pred - self.exploration_coeff*rhs
			
			if self.objective.func_name =="Heat_1" or self.objective.func_name =="Heat_2" or self.objective.func_name =="Heat_3":
				laplacian = compute_laplacian(self.model.forward, x_pde)
				pinn_l = self.exploration_coeff*laplacian
			
			if self.objective.func_name == "BeamDeflection":
				f_prime = autograd.grad(y_pred, x_pde, torch.ones_like(y_pred).to(self.device), retain_graph=True, create_graph=True)[0]
				f_double_prime = autograd.grad(f_prime, x_pde, torch.ones_like(f_prime).to(self.device), retain_graph=True, create_graph=True)[0]
				rho_x = 2.4*x_pde - 64*(torch.pi**2)*torch.exp(4*x_pde)*torch.sin(4*torch.pi*torch.exp(2*x_pde))\
					- 396*torch.exp(2*x_pde)*torch.sin(20*x_pde) + 80*torch.exp(2*x_pde)*torch.cos(20*x_pde) \
						+ 16*torch.pi*torch.exp(2*x_pde)*torch.cos(4*torch.pi*torch.exp(2*x_pde)) + 0.4
				EI_x = torch.exp(x_pde)/rho_x
				f_double_prime = f_double_prime*EI_x
				f_triple_prime = autograd.grad(f_double_prime, x_pde, torch.ones_like(f_double_prime).to(self.device), retain_graph=True, create_graph=True)[0]
				f_fourth_prime = autograd.grad(f_triple_prime, x_pde, torch.ones_like(f_triple_prime).to(self.device))[0]
				pinn_l = self.exploration_coeff*f_fourth_prime - torch.exp(x_pde)
			
			return nn.MSELoss()(pinn_l, torch.zeros_like(pinn_l)).double()

		"""Train neural approximator.
		"""
		x_train = x_train.double()
		y_train = y_train.double()

		self.model.train()
		
		residual_loss = torch.tensor(0)
		total_loss = torch.tensor(0)
		nn_loss = torch.tensor(0)
		print(f"** Train PINN_BO with {epochs} epochs")
		
		alpha_r = 1
		alpha_b = 1
		for i in range(epochs):
			residual_loss = pinn_loss()
			y_pred = self.predict(x_train).double()
			nn_loss = nn.MSELoss()(y_pred, y_train).double()
			total_loss = alpha_b*nn_loss + alpha_r*residual_loss
			self.optimizer.zero_grad()
			total_loss.backward()
			self.optimizer.step()
		return (total_loss, nn_loss, residual_loss)
	
	def optimize_acquisition(self, X, bounds):

		if self.acqf_optimizer == 'Adam':
			adam_optimizer = torch.optim.Adam([X], lr=self.acqf_lr)

			for i in range(self.acqf_epochs):
				adam_optimizer.zero_grad()
				values = self.predict(X)
				values.requires_grad = True
				objective = values.sum()
				objective.backward()
				adam_optimizer.step()
				for j, (lb, ub) in enumerate(bounds):
					X.data[...,j].clamp_(lb, ub)		
		samples = self.predict(X) 

		return X, samples

	def minimize(self, objective):
		torch.manual_seed(0)
		init_points = objective.generate_samples(self.n_init)
		print("Initial points:", init_points)

		self.X_train = init_points['features'].to(self.device)
		self.Y_train = init_points['observations'].to(self.device)
		torch.manual_seed(0)
		X_PDE = objective.generate_features(self.nr).to(self.device)

		# Pick the last point of init_points to draw plot
		if self.objective_type =='synthetic':
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
	


		
		
		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)
		X_mean = (objective.max + objective.min)/2
		X_std = torch.abs(objective.max  - X_mean)


		
		if self.normalized_inputs:
			X_PDE = (X_PDE - X_mean)/X_std

		
		
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
			
			self.train(X_train, Y_train, X_PDE, epochs=self.epochs)
		
		self.model.eval()
		torch.manual_seed(1504)

		x_test = objective.generate_features(10000).to(self.device)
		y_gt = torch.Tensor([objective.value(x, is_noise=False) for x in x_test]).to(self.device)
		if self.normalized_inputs:
			x_test = (x_test- X_mean)/X_std

		Y_pred = self.model(x_test)
		curr_eval = nn.MSELoss()(y_gt, Y_pred.squeeze()).double()
		print(f"Eval MSE at step 0", curr_eval)


		for T in range(self.n_iter):
			

			print(f"----------PINN-BO - Optimization iter {T+1}/{self.n_iter}----------")
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
			Y_cand = self.model(X_cand)
			
			if self.use_local_optimizer:
				_, topk_indices = torch.topk(Y_cand.squeeze(1), self.n_restart, largest=False)
				x_start = X_cand[topk_indices]
				if self.normalized_inputs:
					lb = (objective.min - X_mean)/X_std
					ub = (objective.max - X_mean)/X_std
				else:
					lb = objective.min.to(self.device)
					ub = objective.max.to(self.device)
				x_start.requires_grad = True
				bounds = [(lb[k], ub[k]) for k in range(x_start.shape[1])]
				x, samples_y = self.optimize_acquisition(X=x_start, bounds= bounds)
				min_idx = torch.argmin(samples_y)
				X_next = x[min_idx]
				acq_value = samples_y[min_idx]
			else:
				min_idx= torch.argmin(Y_cand)
				X_next = X_cand[min_idx]
				acq_value = Y_cand[min_idx]
			
			if self.normalized_inputs:
				X_next = (X_next*X_std+X_mean).detach()
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
				if self.normalized_inputs:
					X_train = (self.X_train - X_mean)/X_std
				else:
					X_train = self.X_train

				if self.normalized_outputs:
					Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
				else:
					Y_train = self.Y_train

				self.train(X_train, Y_train, X_PDE, epochs=self.epochs)
				

				self.model.eval()
				
				Y_pred = self.predict(x_test)
				curr_eval = nn.MSELoss()(y_gt, Y_pred.squeeze()).double()
				print(f"Eval MSE at step {T+1}, {curr_eval.item()}")
						
			if (T+1)%(self.n_iter//10) ==0 and self.scheduler !=None:
				self.scheduler.step()

			print(f"** Iter [{T+1}/{self.n_iter}], current value = {true_value}, acq value: {acq_value}")
			optimal_values.append(true_value.item())
		return optimal_values





	