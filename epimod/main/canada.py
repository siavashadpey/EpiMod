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
import epimod.data as data_fetcher

class CachedSEIRSimulation(CachedSimulation):
	def _update_parameters(self):
		(beta, sigma, gamma, kappa) = np.array(self._eqn_parameters, dtype=np.float64)
		eqn = self._ode_solver.equation
		eqn.beta = beta
		eqn.sigma = sigma
		eqn.gamma = gamma
		eqn.kappa = kappa

class RKSolverSeir(RKSolver):
	def __init__(self, ti, tf, n_steps = 1):
		super().__init__(ti, tf, n_steps)

	def get_outputs(self):
		outs = super().get_outputs()
		if self._is_output_stored:
			if self._is_output_grad_needed:
				return (self._time_array, np.diff(outs[1], prepend=0), np.diff(outs[2], prepend=0))
			else:
				return (self._time_array, np.diff(outs[1], prepend=0))
		else:
			error("output is not stored")


def main():

	# observed data
	folder = "./data/"
	region = "canada"
	(t_obs, y_obs, n_pop, shutdown_day, _) = data_fetcher.read_region_data(folder, region)
	y_obs = y_obs/n_pop

	# set eqn
	sigma0 = 1./5.2
	gamma0 = 1./2.28
	beta0 = 2.30*gamma0
	kappa0 = 0.4

	eqn = Seir()
	eqn.tau = shutdown_day
	# TODO: comment out
	eqn.beta = beta0
	eqn.sigma = sigma0
	eqn.gamma = gamma0
	eqn.kappa = kappa0
	
	# set ode solver
	ti = 0
	tf = t_obs[-1]
	n_steps = tf
	rk = RKSolver(ti, tf, n_steps)
	
	rk.output_frequency = 1
	rk.set_output_storing_flag(True)
	rk.equation = eqn
	
	u0 = np.array([1 - y_obs[0], 0, y_obs[0], 0])
	# TODO: uncomment
	#du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
	#rk.set_initial_condition(u0, du0_dp)
	#rk.set_output_gradient_flag(True)

	# TODO: comment out
	rk.set_initial_condition(u0)
	rk.set_output_gradient_flag(False)
	rk.solve()
	(_, y_sim) = rk.get_outputs()

	# TODO: comment out
	plt.plot(t_obs, y_obs*n_pop, 'x', color='b')
	plt.plot(t_obs, y_sim[0,:]*n_pop, color='b', lw=2)

	## sample posterior
	#with pm.Model() as model:
		## set prior distributions
		## TODO: choose accurate priors
		#beta  = pm.Uniform('beta',  lower = 0.7*beta0 , upper = 1.3*beta0 )
		#sigma = pm.Uniform('sigma', lower = 0.7*sigma0, upper = 1.3*sigma0)
		#gamma = pm.Uniform('gamma', lower = 0.7*gamma0, upper = 1.3*gamma0)
		#kappa = pm.Uniform('kappa', lower = 0.5*kappa0, upper = 1.5*kappa0)
		##sigma_normal = pm.Uniform('sigma_normal', lower = 0, upper = 0.1)
	
		## set cached_sim object
		#cached_sim = CachedSEIRSimulation(rk)
	
		## set theano model op object
		#model = ModelOp(cached_sim)
	
		## set likelihood distribution
		##y_sim = pm.Normal('y_sim', mu=model((beta, sigma, gamma, kappa)), sigma=sigma_normal, observed=y_obs)
		##y_sim = pm.NegativeBinomial('y_sim', mu=model((beta, sigma, gamma)), alpha= observed=y_obs)
		#y_sim = pm.Poisson('y_sim', mu=model((beta, sigma, gamma, kappa)), observed=y_obs)
	
		## sample posterior distribution
		#trace = pm.sample(draws=1500, tune=500, cores=5) # using NUTS sampling
	
		## plot posterior distributions of all parameters
		##cached_sim.set_gradient_flag(False)
		#ppc = pm.sample_posterior_predictive(trace)
		#data = az.from_pymc3(trace=trace, posterior_predictive=ppc)
		#az.plot_posterior(data,  credible_interval = 0.95, figsize=(13,3))
		#plt.savefig("fig_dist.pdf")
	
		## propagate uncertainty and make future predictions
		#rk.final_time = tf + 7
		#rk.n_steps = tf + 7
		#ppc_pred = pm.sample_posterior_predictive(trace)
		#ppc_samples = ppc_pred['y_sim']
		#mean_ppc = ppc_samples.mean(axis=0)
		#CriL_ppc = np.percentile(ppc_samples,q=2.5,axis=0)
		#CriU_ppc = np.percentile(ppc_samples,q=97.5,axis=0)

	## plot t vs y_obs
	#fig = plt.figure(figsize=(7, 7))
	#ax = fig.add_subplot(111, xlabel='x', ylabel='y')
	#ax.plot(t_obs, y_obs, 'x', label='sampled data')
	
	## plot propagated uncertainty
	#t_obs_ext = np.linspace(0,rk.final_time,num = rk.final_time + 1)
	#plt.plot(t_obs_ext, mean_ppc[0,:], color='g', lw=2, label='mean of ppc')
	#plt.plot(t_obs_ext, CriL_ppc[0,:], '--', lw=2, color='g', label='95% credible interval')
	#plt.plot(t_obs_ext, CriU_ppc[0,:], '--', lw=2, color='g',)
	#plt.legend(loc=0)
	#plt.savefig("fig_sim.pdf")
	
	plt.show()

if __name__ == '__main__':
	main()