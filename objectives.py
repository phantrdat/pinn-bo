from abc import abstractmethod
from stringprep import c22_specials
import numpy as np
import torch
from torch.distributions.uniform  import Uniform
from numpy.core.records import array
import torch
from pde import CartesianGrid, solve_laplace_equation

class Objective():
	def __init__(self, dim, **kwargs) -> None:
		self.dim = dim
		self.min = torch.full([dim],0)
		self.max = torch.full([dim],1)

		if 'noise_std' in kwargs:
			self.noise_std = kwargs['noise_std']
		else:
			self.noise_std = None
		self.features = None
		self.observations = None
	def noise_std_estimate(self):
		torch.manual_seed(2609)
		points = self.generate_features(10000)
		v = torch.FloatTensor([self.value(x, is_noise=False) for x in points])
		return 0.1*torch.std(v)  # variance of noise equals to 1 percent of function range
	
	def generate_features(self, sample_num):
		# self.features = torch.FloatTensor(sample_num,self.dim).uniform_(self.min ,self.max)
		self.features = torch.rand(sample_num,self.dim).to(self.min.device)
		self.features = torch.mul(self.max-self.min, self.features) + self.min
		return self.features.double()

	def generate_samples(self, sample_num):
		self.generate_features(sample_num)
		self.observations = []
		for x in self.features:
			self.observations.append(self.value(x))
		self.observations = torch.DoubleTensor(self.observations)
		points = {'features': self.features, 'observations':self.observations}
		return points			

	@abstractmethod  
	def value(self, X, is_noise):
		pass
	@abstractmethod
	def constrain(self,X, is_noise):
		pass

	
	@property
	def func_name(self):
		pass
	
class Styblinski_Tang(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim],-5)
		self.max = torch.full([dim], 5)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()
	def value(self, X, is_noise=True):
		if is_noise:
			# noise = torch.normal(mean=0.0, std=torch.tensor(self.noise_std, dtype=torch.float32))
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
		target = 0
		for i in range(self.dim):
			target+= (X[i]**4 - 16*X[i]**2 + 5*X[i])
		target = target/2 + noise
		return target
	@property
	def func_name(self):
		return "Styblinski-Tang"
	
class Cosine_Mixture(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], -1)
		self.max = torch.full([dim], 1)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()
	def value(self, X, is_noise=True):
		if is_noise:
			# noise = torch.normal(mean=0.0, std=torch.tensor(self.noise_std, dtype=torch.float32))
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
		target = 0.1*torch.sum(torch.cos(5*torch.pi*X)) + torch.sum(X**2) + noise
		return target
	@property
	def func_name(self):
		return "Cosine_Mixture"	

class DropWave(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim],-5.12)
		self.max = torch.full([dim], 5.12)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()
	
	def value(self, X, is_noise=True):

		if is_noise:
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
	
		numerator = 1 + torch.cos(12*torch.sqrt(X[0]**2 + X[1]**2))
		denominator = 0.5*(X[0]**2 + X[1]**2)+ 2
		target = -numerator/denominator + noise
		return target
	@property
	def func_name(self):
		return "DropWave"

class Michalewics(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], 0)
		self.max = torch.full([dim], torch.pi)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()
	def value(self, X, is_noise=True):
		if is_noise:
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
		m = 10
		target = 0
		for i in range(self.dim):
			target +=torch.sin(X[i]) * (torch.sin(((i+1)*X[i]*X[i])/torch.pi))**(2*m)

		target = -target + noise
		return target
	@property
	def func_name(self):
		return "Michalewics"

class Rastrigin(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], -5.12)
		self.max = torch.full([dim], 5.12)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()

	def value(self, X, is_noise=True):
		if is_noise:
			# noise = torch.normal(mean=0.0, std=torch.tensor(self.noise_var, dtype=torch.float32))
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
		target = 10*self.dim

		for i in range(self.dim):
			target +=(X[i]**2 - 10*torch.cos(2*torch.pi*X[i]))
		target = target + noise
		return target
	
	@property
	def func_name(self):
		return "Rastrigin"

class BeamDeflection(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], 0)
		self.max = torch.full([dim], 1)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()

	def value(self, x, is_noise=False):
		target = torch.sin(20 * x)*torch.exp(2*x) + torch.sin(4*torch.pi*torch.exp(2*x)) + 0.4 * x**3 + 0.2 * x**2 +6
		return target 
	
	@property
	def func_name(self):
		return "BeamDeflection"

class Heat1(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], 0)
		self.max = torch.full([dim], 2*torch.pi)
		self.noise_std = torch.tensor(0)
		grid = CartesianGrid([[0, 2*np.pi]] * 2, 1000)

		##### Test 1
		bc_x_lower = {"value": "5*sin(y) + sqrt(1+y)"}
		bc_x_upper = {"value":"y*sin(3*cos(y) + 2*exp(y)*sin(y))"}
		bc_y_lower = {"value": "10*cos(x) + x*exp(sqrt(x**2 + sin(x)))"}
		bc_y_upper = {"value": "3*sqrt(exp(x*exp(-x)))*sin(x) + cos(3*x)*cos(3*x)"}

		bcs = [(bc_x_lower, bc_x_upper) , (bc_y_lower, bc_y_upper)]
		res = solve_laplace_equation(grid, bcs)
		m = np.mean(res.data)
		self.f = res.make_interpolator(fill=m)
	def value(self, X, is_noise=False):
		X = X.cpu().numpy()
		return -torch.Tensor(self.f(X))
	@property
	def func_name(self):
		return "Heat_1"
	
class Heat2(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], 0)
		self.max = torch.full([dim], 2*torch.pi)
		self.noise_std = torch.tensor(0)
		grid = CartesianGrid([[0, 2*np.pi]] * 2, 1000)

		bc_x_lower = {"value": "sqrt(2)*sqrt(y)*sin(y) + y**3*cos(2*y) + exp(cos(y))"}
		bc_x_upper = {"value": "sqrt(2)*y**(7/2) + exp(sin(y)) + sin(y)*cos(2*y)"}
		bc_y_lower = {"value": "sqrt(3)*x**(5/2) + exp(sin(x)) + sin(x)*cos(2*x)"}
		bc_y_upper = {"value": "sqrt(3)*sqrt(x)*exp(sin(x)) + x**2*sin(x)**2*cos(x) + exp(cos(x))"}

		bcs = [(bc_x_lower, bc_x_upper) , (bc_y_lower, bc_y_upper)]
		res = solve_laplace_equation(grid, bcs)
		m = np.mean(res.data)
		self.f = res.make_interpolator(fill=m)
	def value(self, X, is_noise=False):
		X = X.cpu().numpy()
		return -torch.Tensor(self.f(X))
	@property
	def func_name(self):
		return "Heat_2"

class Heat3(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim], 0)
		self.max = torch.full([dim], 2*torch.pi)
		self.noise_std = torch.tensor(0)
		grid = CartesianGrid([[0, 2*np.pi]] * 2, 1000)

		bc_x_lower = {"value": "(y**3 + cos(2*y))*(sqrt(2)*sqrt(y) + sin(y)) + exp(cos(y))"}
		bc_x_upper = {"value": "(sqrt(2)*sqrt(y) + y**3)*(sin(y) + cos(2*y)) + exp(sin(y))"}
		bc_y_lower = {"value": "sqrt(3)*sqrt(x)*(sin(x) + cos(2*x)) + x**2 + exp(sin(x))"}
		bc_y_upper = {"value": "(x**2 + sin(x)**2)*exp(cos(x)) + (sqrt(3)*sqrt(x) + exp(sin(x)))*cos(x)"}

		bcs = [(bc_x_lower, bc_x_upper) , (bc_y_lower, bc_y_upper)]
		res = solve_laplace_equation(grid, bcs)
		m = np.mean(res.data)
		self.f = res.make_interpolator(fill=m)
	def value(self, X, is_noise=False):
		X = X.cpu().numpy()
		return -torch.Tensor(self.f(X))
	@property
	def func_name(self):
		return "Heat_3"