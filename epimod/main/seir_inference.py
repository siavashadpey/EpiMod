import argparse
import math
import os

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
				#print(outs[1])
				y = np.diff(outs[1], prepend=0)
				y[y<1.E-14] = 1.E-14
				#print(y)
				return (self._time_array, y, np.diff(outs[2], prepend=0))
			else:
				y = np.diff(outs[1], prepend=0)
				y[y<1.E-14] = 1.E-14
				return (self._time_array, y)
		else:
			error("output is not stored")

def run(region, folder, load_trace=False, compute_sim=True):

	print("started ..." + region)

	if not os.path.exists(region):
		os.makedirs(region)

	# observed data
	(t_obs, y_obs, n_pop, shutdown_day, u0, _) = data_fetcher.read_region_data(folder, region)
	y_obs = y_obs.astype(np.float64) #/n_pop
	u0 = u0.astype(np.float64) #/n_pop
	print(y_obs)
	print(u0)
	print(y_obs.shape)
	# set eqn
	serial_period = 7.2
	sigma0 = 1./1.3
	gamma0 = 1./5 #2.28
	#sigma0 = 1./(serial_period - 1./gamma0) #1.9 #5.2
	beta0 = 2.45*gamma0
	kappa0 = 0.02

	eqn = Seir()
	eqn.tau = shutdown_day
	
	# set ode solver
	ti = t_obs[0]
	tf = t_obs[-1]
	n_steps = tf - ti
	rk = RKSolverSeir(ti, tf, n_steps)
	
	rk.output_frequency = 1
	rk.set_output_storing_flag(True)
	rk.equation = eqn
	
	du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
	rk.set_initial_condition(u0, du0_dp)
	rk.set_output_gradient_flag(True)

	# sample posterior
	with pm.Model() as model:
		# set prior distributions
		#beta  = pm.Uniform('beta',  lower = 0.7*beta0 , upper = 1.3*beta0 )
		#sigma = pm.Uniform('sigma', lower = 0.8*sigma0, upper = 1.2*sigma0)
		#gamma = pm.Uniform('gamma', lower = 0.8*gamma0, upper = 1.2*gamma0)
		#kappa = pm.Uniform('kappa', lower = 0.01, upper = 0.2)
		#dispersion = pm.Uniform('dispersion', lower = 0.1, upper = 30.)
		#beta  = pm.Lognormal('beta',  lower = 0.7*beta0 , upper = 1.3*beta0 )
		#sigma = pm.Lognormal('sigma', mu = math.log(0.1), sigma = 0.5)
		#gamma = pm.Lognormal('gamma', mu = math.log(0.02), sigma = 0.5)
		#kappa = pm.Lognormal('kappa', lower = 0.01, upper = 0.2)
		beta  = pm.Uniform('beta',  lower = 0.4/n_pop, upper = .5/n_pop)
		sigma = pm.Uniform('sigma', lower = 0.1, upper = 0.12)
		gamma = pm.Uniform('gamma', lower = 0.1, upper = .12)
		kappa = pm.Uniform('kappa', lower = 0.02, upper = 0.04)
		dispersion = pm.Normal('dispersion', mu = 30., sigma = 10.)
	
		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)
	
		# set theano model op object
		model = ModelOp(cached_sim)
	
		# set likelihood distribution
		y_sim = pm.NegativeBinomial('y_sim', mu=model((beta, sigma, gamma, kappa)), alpha=dispersion, observed=y_obs)
	
		if not load_trace:
			# sample posterior distribution and save trace
			trace = pm.sample(draws=20, tune=10, cores=4, chains=4, nuts_kwargs=dict(target_accept=0.9), init='advi') # using NUTS sampling
			print(trace.get_values('beta'))
			# save trace
			#pm.backends.text.dump(region + os.path.sep, trace)
		else:
			# load trace
			compute_sim = True
			trace = pm.backends.text.load(region + os.path.sep)		
			
		# plot posterior distributions of all parameters
		data = az.from_pymc3(trace=trace)
		az.plot_posterior(data,  hdi_prob = 0.95)
		plt.savefig(region + os.path.sep + "post_dist.pdf")

		if compute_sim:
			rk.set_output_gradient_flag(False)
			rk.final_time = rk.final_time + 7
			rk.n_steps = rk.n_steps + 7

			betas = trace.get_values('beta')
			sigmas = trace.get_values('sigma')
			gammas = trace.get_values('gamma')
			kappas = trace.get_values('kappa')

			n_samples = betas.shape[0]
			y_sims = np.zeros((n_samples, eqn.n_outputs(), rk.n_steps + 1))
			for i in range(n_samples):
				#print(i)
				eqn.beta = betas[i]
				eqn.sigma = sigmas[i]
				eqn.gamma = gammas[i]
				eqn.kappa = kappas[i]

				rk.solve()
				(_, y_sim_i) = rk.get_outputs()
				y_sims[i,:,:] = y_sim_i


			print("here")
	
			# compute mean and 95% credible interval
			mean_ppc = y_sims.mean(axis=0)
			criL_ppc = np.percentile(y_sims,q=2.5,axis=0)
			criU_ppc = np.percentile(y_sims,q=97.5,axis=0)
			print("here2")

			# linear plot
			# plot t vs y_obs
			fig = plt.figure(figsize=(7, 7))
			ax = fig.add_subplot(111, xlabel='x', ylabel='y')
			ax.plot(t_obs, y_obs*n_pop, 'x')
			
			## plot propagated uncertainty
			t_obs_ext = np.linspace(rk.initial_time,rk.final_time,num = rk.n_steps + 1)
			plt.plot(t_obs_ext, mean_ppc[0,:]*n_pop, color='g', lw=2, label='mean')
			plt.plot(t_obs_ext, criL_ppc[0,:]*n_pop, '--', lw=2, color='g', label='95% credible interval')
			plt.plot(t_obs_ext, criU_ppc[0,:]*n_pop, '--', lw=2, color='g')
			plt.legend(loc=0)
		
			plt.savefig(region + os.path.sep + "linear.pdf")

			# log plot
			fig = plt.figure(figsize=(7, 7))
			ax = fig.add_subplot(111, xlabel='x', ylabel='y')
			ax.semilogy(t_obs, y_obs*n_pop, 'x')
			
			## plot propagated uncertainty
			t_obs_ext = np.linspace(rk.initial_time,rk.final_time,num = rk.n_steps + 1)
			plt.semilogy(t_obs_ext, mean_ppc[0,:]*n_pop, color='g', lw=2, label='mean')
			plt.semilogy(t_obs_ext, criL_ppc[0,:]*n_pop, '--', lw=2, color='g', label='95% credible interval')
			plt.semilogy(t_obs_ext, criU_ppc[0,:]*n_pop, '--', lw=2, color='g')
		
			plt.savefig(region + os.path.sep + "log.pdf")
	
	print("finished ..." + region)

def main():
	parser = argparse.ArgumentParser(description="Runs the inference problem for the specified region.")
	parser.add_argument('--folder', '-f', default='./data/', help='path of data folder')
	parser.add_argument('--region', '-r', default='canada', help='region of interest')
	parser.add_argument('--load_trace', '-l', action='store_true', default=False, help='flag indicating to load trace')
	parser.add_argument('--no_propagate', '-np', action='store_true', default=False, help='flag indicating not to perform UQ or not')

	args = parser.parse_args()

	folder = args.folder
	region = args.region.lower()
	load_trace = args.load_trace
	propagate = not args.no_propagate
	run(region, folder, load_trace, propagate)

if __name__ == '__main__':
	main()