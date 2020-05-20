import numpy as np
from equation import Equation

class Seir(Equation):
	def __init__(self, beta=0, sigma=0, gamma=0):
		super().__init__()
		self._n_components = 4
		self._n_parameters = 3
		self._n_outputs = 1
		self._beta = beta
		self._sigma = sigma
		self._gamma = gamma

	@property
	def beta(self):
		return self._beta

	@beta.setter
	def beta(self, value):
		self._beta = value

	@property
	def sigma(self):
		return self._sigma

	@sigma.setter
	def sigma(self, value):
		self._sigma = value

	@property
	def gamma(self):
		return self._gamma

	@gamma.setter
	def gamma(self, value):
		self._gamma = gamma

	def source(self, t, u, is_grad_needed = False):

		assert u.shape == (self._n_components,), "u is not a vector of size n_components"
		
		S = u[0]
		E = u[1]
		I = u[2]
		# R = u[3] # Not used

		b = self._beta
		s = self._sigma
		g = self._gamma

		f = np.array([
			-b*S*I,
			 b*S*I - s*E,
			 s*E - g*I, 
			 g*I])

		if not is_grad_needed:
			return f

		df_du = np.array([[-b*I,  0, -b*S, 0],
						  [ b*I, -s,  b*S, 0],
						  [   0,  s,   -g, 0],
						  [   0,  0,    g, 0]])

		df_dp = np.array([[-S*I,  0,  0],
						  [ S*I, -E,  0],
						  [   0,  E, -I],
						  [   0,  0,  I]])

		return (f, df_du, df_dp)

	def output(self, t, u, du_dp = None):

		assert u.shape == (self._n_components,) , "u is not a vector of size n_components"

		if du_dp is None:
			return u[2]

		assert du_dp.shape == (self._n_components, self._n_parameters), "du_dp is not of size n_components x n_parameters"
		return (u[2], du_dp[2,:])

eqn = Seir()
print(eqn.beta)