import argparse
import math
import os
import datetime as dt

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
import epimod.data.read_region_data as data_fetcher

class CachedSEIRSimulation(CachedSimulation):
	def _update_parameters(self):
		(beta, sigma, gamma, kappa, tint) = np.array(self._eqn_parameters, dtype=np.float64)
		eqn = self._ode_solver.equation
		eqn.beta = beta
		eqn.sigma = sigma
		eqn.gamma = gamma
		eqn.kappa = kappa
		eqn.tint = tint

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

	print("started ... " + region)

	if not os.path.exists(region):
		os.makedirs(region)

	# observed data
	(t_obs, datetimes, y_obs, n_pop, shutdown_day, u0, _) = data_fetcher.read_region_data(folder, region)
	y_obs = y_obs.astype(np.float64)
	u0 = u0.astype(np.float64)
	
	# set eqn
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
		beta  = pm.Lognormal('beta',  mu = math.log(0.3/n_pop), sigma = 0.5)
		sigma = pm.Lognormal('sigma', mu = math.log(0.05), sigma = 0.6)
		gamma = pm.Lognormal('gamma', mu = math.log(0.05), sigma = 0.6)
		kappa = pm.Lognormal('kappa', mu = math.log(0.001), sigma = 0.8)
		tint = pm.Lognormal('tint', mu = math.log(30), sigma = math.log(10))
		dispersion = pm.Normal('dispersion', mu = 30., sigma = 10.)
	
		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)
	
		# set theano model op object
		model = ModelOp(cached_sim)
	
		# set likelihood distribution
		y_sim = pm.NegativeBinomial('y_sim', mu=model((beta, sigma, gamma, kappa, tint)), alpha=dispersion, observed=y_obs)
		
		if not load_trace:
			# sample posterior distribution and save trace
			trace = pm.sample(draws=20, tune=10, cores=4, chains=4, nuts_kwargs=dict(target_accept=0.9), init='advi+adapt_diag') # using NUTS sampling
			# save trace
			pm.backends.text.dump(region + os.path.sep, trace)
		else:
			# load trace
			trace = pm.backends.text.load(region + os.path.sep)		
		
		# plot posterior distributions of all parameters
		data = az.from_pymc3(trace=trace)
		pm.plots.traceplot(data, legend=True)
		plt.savefig(region + os.path.sep + "trace_plot.pdf")
		az.plot_posterior(data,  hdi_prob = 0.95)
		plt.savefig(region + os.path.sep + "post_dist.pdf")

		if compute_sim:
			rk.set_output_gradient_flag(False)
			n_predictions = 7
			rk.final_time = rk.final_time + n_predictions
			rk.n_steps = rk.n_steps + n_predictions

			betas = trace.get_values('beta')
			sigmas = trace.get_values('sigma')
			gammas = trace.get_values('gamma')
			kappas = trace.get_values('kappa')
			tints = trace.get_values('tint')

			n_samples = betas.shape[0]
			y_sims = np.zeros((n_samples, eqn.n_outputs(), rk.n_steps + 1))
			# TODO: parallelize
			for i in range(n_samples):
				eqn.beta = betas[i]
				eqn.sigma = sigmas[i]
				eqn.gamma = gammas[i]
				eqn.kappa = kappas[i]
				eqn.tint = tints[i]

				rk.solve()
				(_, y_sim_i) = rk.get_outputs()
				y_sims[i,:,:] = y_sim_i
	
			# compute mean and 95% credible interval
			mean_y = np.mean(y_sims[:,0,:], axis=0)
			lower_y = np.percentile(y_sims[:,0,:],q=2.5,axis=0)
			upper_y = np.percentile(y_sims[:,0,:],q=97.5,axis=0)

			# plots
			dates = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in datetimes]
			pred_dates = dates + [dates[-1] + dt.timedelta(days=i) for i in range(1,1 + n_predictions)]

			# linear plot
			fig = plt.figure(figsize=(7, 7))
			ax = fig.add_subplot(111, xlabel='x', ylabel='y')
			ax.plot(dates, y_obs, 'x', color='k')
			import matplotlib.dates as mdates
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
			plt.title(region[0].upper() + region[1:].lower() + "'s daily infections")
			plt.xlabel('Date')
			plt.ylabel('New daily infections')

			# plot propagated uncertainty
			plt.plot(pred_dates, mean_y, color='g', lw=2, label='mean')
			plt.fill_between(pred_dates, lower_y, upper_y, color='darkseagreen', label='95% credible interval')
			#plt.plot(
			#plt.plot(pred_dates, upper_y, '--', lw=2, color='g')
			plt.legend(loc='upper left')
			fig.autofmt_xdate()
			plt.savefig(region + os.path.sep + "linear.pdf")

			# log plot
			plt.yscale('log')
			plt.savefig(region + os.path.sep + "log.pdf")
	
	print("finished ... " + region)

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