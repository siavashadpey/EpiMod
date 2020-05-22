import math
import numpy as np
import matplotlib.pyplot as plt

import theano as theano
import theano.tensor as tt

import pymc3 as pm
import arviz as az

from epimod.solver.ode_solver.rk_solver import RKSolver
from epimod.eqn.equation import Equation
from epimod.eqn.seir import Seir

theano.config.exception_verbosity= 'high'

class CachedSimulation:
	def __init__(self, ode_solver):
		self._ode_solver = ode_solver
		self._cached_output = None
		self._cached_doutput_dparams = None
		self._cached_beta = None
		self._cached_sigma = None
		self._cached_gamma = None
		self._grad_flag = True

	def __call__(self, beta, sigma, gamma):
		#print(beta)
		#print(sigma)
		#print(gamma)
		#print("")
		if not self._is_cached(beta, sigma, gamma):
			#print(beta)
			#print(sigma)
			#print(gamma)
			self._ode_solver.equation.beta  = beta
			self._ode_solver.equation.sigma = sigma
			self._ode_solver.equation.gamma = gamma
			self._ode_solver.set_output_gradient_flag(self._grad_flag)
			self._ode_solver.solve()
			if self._grad_flag:
				(t, self._cached_output, self._cached_doutput_dparams) = self._ode_solver.get_outputs()
			else:
				(t, self._cached_output) = self._ode_solver.get_outputs()
				self._cached_doutput_dparams = None
			#(t, self._cached_output) = self._ode_solver.get_outputs()
			self._cached_beta = beta
			self._cached_sigma = sigma
			self._cached_gamma = gamma

		return (self._cached_output, self._cached_doutput_dparams)

	def set_gradient_flag(self, flag):
		self._grad_flag = flag

	def _is_cached(self, beta, sigma, gamma):
		return (self._cached_beta == beta  and
			   	self._cached_sigma == sigma and
			    self._cached_gamma == gamma)

class SimModelGradOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim

	def make_node(self, beta, sigma, gamma, dL_df):
		beta = tt.as_tensor_variable(beta)
		sigma = tt.as_tensor_variable(sigma)
		gamma = tt.as_tensor_variable(gamma)
		dL_df = tt.as_tensor_variable(dL_df)
		inputs = [beta, sigma, gamma, dL_df]
		outputs = [(theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(beta.dtype),
                      broadcastable=[False] * 1)())]
		node = theano.Apply(op = self, inputs = inputs, outputs = outputs)
		return node

	def perform(self, node, inputs, outputs):
		(beta, sigma, gamma, dL_df) = inputs
		df_dparams = self._cached_sim(beta, sigma, gamma)[1]
		x = dL_df[0].dot(df_dparams[:,0].T)
		y = dL_df[0].dot(df_dparams[:,1].T)
		z = dL_df[0].dot(df_dparams[:,2].T)
		outputs[0][0] = [x.item(), y.item(), z.item()]

class SimModelOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim
		self._grad_op = SimModelGradOp(cached_sim)

	def make_node(self, beta, sigma, gamma):
		beta = tt.as_tensor_variable(beta)
		sigma = tt.as_tensor_variable(sigma)
		gamma = tt.as_tensor_variable(gamma)
		inputs = [beta, sigma, gamma]
		outputs = [(theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(beta.dtype),
                      broadcastable=[False] * 2)())]
		node = theano.Apply(op = self, inputs = inputs, outputs = outputs)
		return node

	def perform(self, node, inputs, outputs):
		(beta, sigma, gamma) = inputs
		outputs[0][0] = self._cached_sim(beta, sigma, gamma)[0]

	def grad(self, inputs, output_grads):
		(beta, sigma, gamma) = inputs
		grad = self._grad_op(beta, sigma, gamma, output_grads)
		return [grad[0], grad[1], grad[2]]

def fetch_observed_data():
	n_pop = 7E6
	sigma = 1./5.2 # 0.19
	gamma = 1./2.28 # 0.43
	beta = 2.13*gamma # 0.93
	eqn = Seir(beta, sigma, gamma)

	ti = 0
	tf = 10
	n_steps = tf
	rk = RKSolver(ti, tf, n_steps)
	rk.output_frequency = 1
	rk.set_output_storing_flag(True)

	rk.equation = eqn

	u0 = np.array([n_pop - 1, 0, 1, 0])
	u0 /= n_pop
	rk.set_initial_condition(u0)

	rk.solve()

	(t, y_sim) = rk.get_outputs()

	# perturb output using a normal distribution
	mu = y_sim[0,:]
	sigma = y_sim[0,1] # 1E-7
	y_obs = np.random.normal(mu , sigma)
	return (t, y_obs)

def main():

	# artificially generate observed data
	(t_obs, y_obs) = fetch_observed_data()

	# sample posterior
	with pm.Model() as model:
		# set prior distributions
		beta = pm.Uniform('beta', lower = 0.5, upper = 1.5)
		sigma = pm.Uniform('sigma', lower = 0.05, upper = 0.5)
		gamma = pm.Uniform('gamma', lower = 0.2, upper = 0.8)
		sigma_normal = pm.Uniform('sigma_normal', lower = 0, upper = 0.1)

		# set eqn
		eqn = Seir()

		# set ode solver
		ti = 0
		tf = 10
		n_steps = tf
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)
		rk.set_output_gradient_flag(False)

		# set cached_sim object
		cached_sim = CachedSimulation(rk)

		# set theano model op object
		model = SimModelOp(cached_sim)

		# set likelihood distribution
		y_sim = pm.Normal('y_sim', mu=model(beta, sigma, gamma), sigma=sigma_normal, observed=y_obs)

		# sample posterior distributions
		trace = pm.sample(draws=20, tune=10, cores=2) # using NUTS sampling

		# plot posterior distributions of all parameters
		#cached_sim.set_gradient_flag(False)
		ppc = pm.sample_posterior_predictive(trace)
		data = az.from_pymc3(trace=trace, posterior_predictive=ppc)
		az.plot_posterior(data,  credible_interval = 0.95, figsize=(13,3))
		plt.savefig("fig_dist.pdf")

		# propagate uncertainty and make future predictions
		rk.final_time = tf + 3
		rk.n_steps = tf + 3
		ppc_pred = pm.sample_posterior_predictive(trace)
		ppc_samples = ppc_pred['y_sim']
		mean_ppc = ppc_samples.mean(axis=0)
		CriL_ppc = np.percentile(ppc_samples,q=2.5,axis=0)
		CriU_ppc = np.percentile(ppc_samples,q=97.5,axis=0)

	# plot t vs y_obs
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111, xlabel='x', ylabel='y')
	ax.semilogy(t_obs, y_obs, 'x', label='sampled data')
	
	# plot propagated uncertainty
	t_obs_ext = np.linspace(0,rk.final_time,num = rk.final_time + 1)
	plt.semilogy(t_obs_ext, mean_ppc[0,:], color='g', lw=2, label='mean of ppc')
	plt.semilogy(t_obs_ext, CriL_ppc[0,:], '--', lw=2, color='g', label='95% credible interval')
	plt.semilogy(t_obs_ext, CriU_ppc[0,:], '--', lw=2, color='g',)
	plt.legend(loc=0)
	plt.savefig("fig_sim.pdf")

	#plt.show()

if __name__ == '__main__':
	main()