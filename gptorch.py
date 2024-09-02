from curses import raw
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch import fit_gpytorch_mll
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, NoisyExpectedImprovement, AnalyticAcquisitionFunction
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
import torch
import time, math
from torch.quasirandom import SobolEngine
from botorch.utils.sampling import draw_sobol_samples
import warnings
warnings.filterwarnings("ignore")

GP_common_kernels = {'Matern': ScaleKernel(MaternKernel(nu=0.5)), 'RBF': ScaleKernel(RBFKernel())}



class GPTorch(object):
	def __init__(self, cfg):

		self.kernel_name = cfg.kernel_name
		self.n_iter = cfg.n_iter
		self.model = None 
		self.X_train = None
		self.Y_train = None
		self.exploration_coeff =  0
		self.acquisition_name = cfg.acquisition_name
		self.n_restart = cfg.n_restart
		self.n_raw_samples  = cfg.n_raw_samples
		self.n_init = cfg.n_init
		self.objective_type = cfg.objective_type
		self.device = torch.device(0 if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
		self.normalized_inputs = cfg.normalized_inputs
		self.normalized_outputs = cfg.normalized_outputs
	def minimize(self, objective):

		torch.manual_seed(0)
		init_points = objective.generate_samples(self.n_init)
		print("Initial points:", init_points)
		self.X_train = init_points['features'].to(self.device)
		self.Y_train = init_points['observations'].to(self.device)
		
		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)

		if self.normalized_inputs:
			X_mean = (objective.max + objective.min)/2
			X_std = torch.abs(objective.max  - X_mean) 
			lb = (objective.min - X_mean)/X_std
			ub = (objective.max - X_mean)/X_std
		else:
			lb = objective.min
			ub = objective.max
		bounds = torch.stack([lb, ub]).float().to(self.device)
		
		

		if self.objective_type =='synthetic':
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		else:
			optimal_values = [init_points['observations'][-1].item()]
		
		X_next = None
		for i in range(self.n_iter):
						
			
			frac, whole = math.modf(time.time())
			seed = int(whole/(10000*frac))+i
			torch.manual_seed(seed)
			self.iteration = i+1

			acquisition_func = None
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train
			
			X_train = X_train.double()
			
			Y_train = self.Y_train.unsqueeze(-1).double().to(self.device)
			if self.acquisition_name == 'UCB':
				# self.model = FixedNoiseGP(self.X_train, Y_train, train_Yvar = torch.full_like(Y_train, objective.noise_std**2)).to(self.device)
				self.model = SingleTaskGP(X_train, Y_train).to(self.device)
				mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
				fit_gpytorch_mll(mll)
				radius = torch.abs(torch.max(objective.max - objective.min))
				delta = 0.05
				a = 1
				lengthscale = self.model.covar_module.base_kernel.lengthscale.squeeze(0)[0].item()
				b = 1 / 2 * torch.sqrt(torch.tensor(2)) * 1 / lengthscale
				beta_t = (4 * objective.dim + 4) * torch.log(torch.tensor(self.n_iter)) + 2 * torch.log(torch.tensor(2 * torch.pi ** 2 / (3 * delta))) \
					+ 2 * objective.dim * torch.log(objective.dim * b * radius * torch.sqrt(torch.log(torch.tensor(4 * objective.dim * a / delta))))
				acquisition_func = UpperConfidenceBound(self.model, beta=beta_t, maximize=False)
			elif self.acquisition_name =='EI':
				if objective.noise_std!=0:
					self.model = FixedNoiseGP(X_train, Y_train, train_Yvar = torch.full_like(Y_train, objective.noise_std**2))
				else:
					self.model = SingleTaskGP(X_train, Y_train).to(self.device)
				mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
				try:
					fit_gpytorch_mll(mll)
				except:
					pass
				# best_f = torch.min(torch.FloatTensor(optimal_values))
				if objective.noise_std !=0:
					acquisition_func = NoisyExpectedImprovement(self.model, self.X_train, num_fantasies=5, maximize=False)
				else:
					best_f = torch.min(torch.FloatTensor(optimal_values))
					acquisition_func = ExpectedImprovement(self.model, best_f=best_f, maximize=False)
			# elif self.acquisition_name =='TS':
			# 	acquisition_func = MaxPosteriorSampling(self.model)

			# if self.acquisition_name =='TS':
			# 	sobol = SobolEngine(self.X_train.shape[-1], scramble=True)
			# 	X_cand = sobol.draw(512).to(dtype=torch.float, device=self.device)
			# 	X_cand = objective.min + (objective.max-objective.min)*X_cand
			# 	# X_cand = objective.generate_features(n_restart)
			# 	gpts = MaxPosteriorSampling(model=self.model, replacement=False)
			# 	X_next = gpts(X_cand, num_samples=1)
			# else:
				# X_next ,_ = optimize_acqf(acq_function=acquisition_func, bounds=bounds, q=1, 
				# 							batch_initial_conditions=X_cand.unsqueeze(1), num_restarts=n_restart)

			try:
				X_next, _ = optimize_acqf(acq_function=acquisition_func, bounds=bounds, q=1, 
														raw_samples=self.n_raw_samples, num_restarts=self.n_restart)
			except:
				pass					

			posterior = self.model(X_next)

			predictive_mean = posterior.mean.item()
			predictive_var = posterior.variance.item()
			
			if self.normalized_inputs:
				X_next = (X_next*X_std + X_mean).detach()
			else:
				X_next = X_next.detach()
			
			observation = objective.value(X_next[0])
			if self.objective_type == 'synthetic':
				true_value = objective.value(X_next[0], is_noise=False)
			else:
				true_value = observation
			if observation.device.type =='cpu':
				observation = observation.to(self.device)

			self.X_train = torch.cat([self.X_train, X_next.clone()])
			if objective.dim >= 2:
				self.Y_train = torch.cat([self.Y_train, observation.unsqueeze(-1)])
			else:
				self.Y_train = torch.cat([self.Y_train, observation])
			print(f"** Round [{i+1}/{self.n_iter}], current value = {true_value},", 
						f"predictive mean {predictive_mean}, predictive var {predictive_var}")
			optimal_values.append(true_value.item())

		return optimal_values







			


	

	