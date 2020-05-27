import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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
	(t_obs, y_obs, n_pop, shutdown_day, u0, _) = data_fetcher.read_region_data(folder, region)
	y_obs = y_obs.astype(np.float64)
	u0 = u0.astype(np.float64)
	#print(shutdown_day)

	# set eqn
	serial_period = 7.2
	sigma0 = 1./1.3
	gamma0 = 1./5 #2.28
	#sigma0 = 1./(serial_period - 1./gamma0) #1.9 #5.2
	beta0 = 2.45*gamma0/n_pop
	kappa0 = 0.02

	eqn = Seir()
	eqn.tau = shutdown_day
	
	# set ode solver
	ti = t_obs[0]
	tf = t_obs[-1]
	n_steps = tf
	rk = RKSolverSeir(ti, tf, n_steps)
	
	rk.output_frequency = 1
	rk.set_output_storing_flag(True)
	rk.equation = eqn
	
	du0_dp = np.zeros(eqn.n_components(), eqn.n_components())
	rk.set_initial_condition(u0, du0_dp)
	rk.set_output_gradient_flag(True)

	# sample posterior
	with pm.Model() as model:
		# set prior distributions
		beta  = pm.Uniform('beta',  lower = 0.7*beta0 , upper = 1.3*beta0 )
		sigma = pm.Uniform('sigma', lower = 0.8*sigma0, upper = 1.2*sigma0)
		gamma = pm.Uniform('gamma', lower = 0.8*gamma0, upper = 1.2*gamma0)
		kappa = pm.Uniform('kappa', lower = 0.01, upper = 0.2)
		dispersion = pm.Uniform('dispersion', lower = 0.1, upper = 30.)
	
		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)
	
		# set theano model op object
		model = ModelOp(cached_sim)
	
		# set likelihood distribution
		y_sim = pm.Normal('y_sim', mu=model((beta, sigma, gamma, kappa)), sigma=dispersion, observed=y_obs)
		#y_sim = pm.NegativeBinomial('y_sim', mu=model((beta, sigma, gamma)), alpha=dispersion, observed=y_obs)
		#y_sim = pm.Poisson('y_sim', mu=model((beta, sigma, gamma, kappa)), observed=y_obs)
	
		## sample posterior distribution
		trace = pm.sample(draws=200, tune=100, cores=2, chains=2, nuts_kwargs=dict(target_accept=0.9), init='adapt_diag') # using NUTS sampling
	
		# plot posterior distributions of all parameters
		#cached_sim.set_gradient_flag(False)
		ppc = pm.sample_posterior_predictive(trace)
		data = az.from_pymc3(trace=trace, posterior_predictive=ppc)
		az.plot_posterior(data,  credible_interval = 0.95)
		plt.savefig("fig_dist.pdf")
	
		## propagate uncertainty and make future predictions
		rk.final_time = tf + 7
		rk.n_steps = tf + 7
		ppc_pred = pm.sample_posterior_predictive(trace)
		ppc_samples = ppc_pred['y_sim']
		mean_ppc = ppc_samples.mean(axis=0)
		CriL_ppc = np.percentile(ppc_samples,q=2.5,axis=0)
		CriU_ppc = np.percentile(ppc_samples,q=97.5,axis=0)

	## plot t vs y_obs
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111, xlabel='x', ylabel='y')
	ax.plot(t_obs, y_obs, 'x', label='sampled data')
	
	## plot propagated uncertainty
	t_obs_ext = np.linspace(0,rk.final_time,num = rk.final_time + 1)
	plt.plot(t_obs_ext, mean_ppc[0,:], color='g', lw=2, label='mean of ppc')
	plt.plot(t_obs_ext, CriL_ppc[0,:], '--', lw=2, color='g', label='95% credible interval')
	plt.plot(t_obs_ext, CriU_ppc[0,:], '--', lw=2, color='g',)
	plt.legend(loc=0)
	plt.savefig("fig_sim.pdf")
	
	#plt.show()

if __name__ == '__main__':
	main()