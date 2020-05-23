import math
import numpy as np
import matplotlib.pyplot as plt

import theano as theano

import pymc3 as pm
import arviz as az

from epimod.model.simulation_operator import CachedSimulation
from epimod.model.simulation_operator import ModelOp
from epimod.model.simulation_operator import ModelGradOp
from epimod.eqn.seir import Seir
from epimod.solver.ode_solver.rk_solver import RKSolver

class CachedSEIRSimulation(CachedSimulation):
	def _update_parameters(self):
		(beta, sigma, gamma) = np.array(self._eqn_parameters, dtype=np.float64)
		eqn = self._ode_solver.equation
		eqn.beta = beta
		eqn.sigma = sigma
		eqn.gamma = gamma

def fetch_observed_data():
	n_pop = 7E6
	sigma = 1./5.2 # 0.19
	gamma = 1./2.28 # 0.43
	beta = 2.13*gamma # 0.93
	eqn = Seir(beta, sigma, gamma)

	ti = 0
	tf = 20
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
		tf = 20
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
		rk.set_output_gradient_flag(True)

		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)

		# set theano model op object
		model = ModelOp(cached_sim)

		# set likelihood distribution
		y_sim = pm.Normal('y_sim', mu=model((beta, sigma, gamma)), sigma=sigma_normal, observed=y_obs)

		# sample posterior distributions
		trace = pm.sample(draws=200, tune=100, cores=2) # using NUTS sampling

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

	plt.show()

if __name__ == '__main__':
	main()