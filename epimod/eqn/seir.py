import math

import numpy as np

from epimod.eqn.equation import Equation

class Seir(Equation):
	def __init__(self, beta=0, sigma=0, gamma=0, kappa = 1, tau = math.inf):
		super().__init__()
		self._n_components = 4
		self._n_parameters = 4
		self._n_outputs = 1
		self._beta0 = beta
		self._sigma = sigma
		self._gamma = gamma
		self._kappa = kappa
		self._tau = tau

	@property
	def beta(self):
		#TODO: output the actual time-varying beta
		return self._beta0

	@beta.setter
	def beta(self, value):
		self._beta0 = value # beta0: transmissibility rate

	@property
	def sigma(self):
		return self._sigma # 1/sigma: latent period (period between being exposed and bein infectious)
		# incubation period is period between being exposed and showing symptoms
		# however, a person can be infectious before showing symptoms
		# i.e. latent period could be shorter than incubation period

	@sigma.setter
	def sigma(self, value):
		self._sigma = value

	@property
	def gamma(self):
		return self._gamma #1/gamma: infectious period

	@gamma.setter
	def gamma(self, value):
		self._gamma = value

	@property
	def tau(self):
		return self._tau # day when social distancing measures are introduced
	
	@tau.setter
	def tau(self, value):
		self._tau = value
	
	@property
	def kappa(self):
		return self._kappa

	@kappa.setter
	def kappa(self, value):
		self._kappa = value # represents social distancing measures effectiveness

	def source(self, t = 0, u = 0, is_grad_needed = False):

		assert u.shape == (self._n_components,), "u is not a vector of size n_components"
		
		S = u[0]
		E = u[1]
		I = u[2]
		# R = u[3] # Not used

		(b, db0, dkappa) = Seir._compute_beta(self._beta0, self._kappa, self._tau, t)
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

		# rows: df_dbeta0, df_dsigma, df_dgamma, df_dkappa
		df_dp = np.array([[-db0*S*I,  0,  0, -dkappa*S*I],
						  [ db0*S*I, -E,  0,  dkappa*S*I],
						  [       0,  E, -I,           0],
						  [       0,  0,  I,           0]])

		return (f, df_du, df_dp)

	def output(self, t = 0, u = 0, du_dp = None):

		assert u.shape == (self._n_components,) , "u is not a vector of size n_components"

		if du_dp is None:
			return u[2]

		assert du_dp.shape == (self._n_components, self._n_parameters), "du_dp is not of size n_components x n_parameters"
		return (u[2], du_dp[2,:])

	@staticmethod
	def _compute_beta(beta0, kappa, tau, t):
		dt = t - tau
		if dt <= 0:
			return (beta0, 1, 0)
		else:
			exp = math.exp(-kappa*dt)
			return (beta0*exp, exp, -beta0*exp*dt)

