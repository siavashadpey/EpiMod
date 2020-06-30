import argparse
import math
import os
import datetime as dt

import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pymc3 as pm
import arviz as az
import pandas as pd

from epimod.model.simulation_operator import CachedSimulation
from epimod.model.simulation_operator import ModelOp
from epimod.model.simulation_operator import ModelGradOp
from epimod.eqn.seir import Seir
from epimod.solver.ode_solver.rk_solver import RKSolver
import epimod.data.read_region_data as data_fetcher

class CachedSEIRSimulation(CachedSimulation):
    def _update_parameters(self):
        (beta, sigma, gamma, kappa, tint) = self._eqn_parameters
        #print(beta, sigma, gamma, kappa, tint)
        import sys
        sys.stdout.flush()
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
                y = np.diff(outs[1], prepend=0)
                y[y<1.E-14] = 1.E-14
                return (self._time_array, y, np.diff(outs[2], prepend=0))
            else:
                y = np.diff(outs[1], prepend=0)
                y[y<1.E-14] = 1.E-14
                return (self._time_array, y)
        else:
            error("output is not stored")

def dist_from_samples(param_name, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.distributions.Interpolated(param_name, x, y)

def single_simulation(tup, rk):
    (beta, sigma, gamma, kappa, tint) = tup
    rk.equation.beta = beta
    rk.equation.sigma = sigma
    rk.equation.gamma = gamma
    rk.equation.kappa = kappa
    rk.equation.tint = tint
    rk.solve()
    (_, y_sim) = rk.get_outputs()   
    return y_sim

def run(region, folder, smart_prior=False, load_trace=False, compute_sim=True, plot_posterior_dist = True):

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
    m = 2
    n_steps = m*(tf - ti)
    rk = RKSolverSeir(ti, tf, n_steps)
    rk.rk_type = "explicit_euler"
    rk.output_frequency = m
    rk.set_output_storing_flag(True)
    rk.equation = eqn
    
    du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
    rk.set_initial_condition(u0, du0_dp)
    rk.set_output_gradient_flag(True)

    # sample posterior
    with pm.Model() as model:
        # set prior distributions
        #beta  = pm.Lognormal('beta',  mu = math.log(0.4/n_pop), sigma = 0.4)
        #sigma = pm.Lognormal('sigma', mu = math.log(0.3), sigma = 0.5)
        #gamma = pm.Lognormal('gamma', mu = math.log(0.25), sigma = 0.5)
        #kappa = pm.Lognormal('kappa', mu = math.log(0.1), sigma = 0.5)

        #beta  = pm.Normal('beta',  mu = 0.4/n_pop, sigma = 0.06/n_pop)
        #sigma = pm.Normal('sigma', mu = 0.6, sigma = 0.1)
        #gamma = pm.Normal('gamma', mu = 0.3, sigma = 0.07)
        #kappa = pm.Normal('kappa', mu = 0.5, sigma = 0.1)
        #tint = pm.Lognormal('tint', mu = math.log(30), sigma = 1)
        if not smart_prior:
            beta  = pm.Lognormal('beta',  mu = math.log(0.3/n_pop), sigma = 0.5)
            sigma = pm.Lognormal('sigma', mu = math.log(0.05), sigma = 0.6)
            gamma = pm.Lognormal('gamma', mu = math.log(0.05), sigma = 0.6)
            kappa = pm.Lognormal('kappa', mu = math.log(0.001), sigma = 0.8)
            tint = pm.Lognormal('tint', mu = math.log(30), sigma = math.log(10))
            dispersion = pm.Normal('dispersion', mu = 30., sigma = 10.)
        else:
            # use old posterior dist as new prior dist
            trace = pm.backends.text.load(region + os.path.sep)
            beta = dist_from_samples('beta', trace['beta'])
            sigma = dist_from_samples('sigma', trace['sigma'])
            gamma = dist_from_samples('gamma', trace['gamma'])
            kappa = dist_from_samples('kappa', trace['kappa'])
            tint = dist_from_samples('tint', trace['tint'])
            dispersion = dist_from_samples('dispersion', trace['dispersion'])
    
        # set cached_sim object
        cached_sim = CachedSEIRSimulation(rk)
    
        # set theano model op object
        model = ModelOp(cached_sim)
    
        # set likelihood distribution
        y_sim = pm.NegativeBinomial('y_sim', mu=model((beta, sigma, gamma, kappa, tint)), alpha=dispersion, observed=y_obs)
        
        if not load_trace:
            # sample posterior distribution and save trace
            draws = 20 #1000
            tune = 10 #500
            trace = pm.sample(draws=draws, tune=tune, cores=4, chains=4, nuts_kwargs=dict(target_accept=0.9), init='advi+adapt_diag') # using NUTS sampling
            # save trace
            pm.backends.text.dump(region + os.path.sep, trace)
        else:
            # load trace
            trace = pm.backends.text.load(region + os.path.sep)     
        
        if plot_posterior_dist:
        # plot posterior distributions of all parameters
            data = az.from_pymc3(trace=trace)
            pm.plots.traceplot(data, legend=True)
            plt.savefig(region + os.path.sep + "trace_plot.pdf")
            az.plot_posterior(data,  hdi_prob = 0.95)
            plt.savefig(region + os.path.sep + "post_dist.pdf")

        if compute_sim:
            #rk.set_output_gradient_flag(False)
            n_predictions = 7
            rk.final_time = rk.final_time + n_predictions
            rk.n_steps = rk.n_steps + m*n_predictions

            y_sims = pm.sample_posterior_predictive(trace)['y_sim'][:,0,:]
            np.savetxt(region + os.path.sep + "y_sims.csv", y_sims, delimiter = ',')
            mean_y = np.mean(y_sims,axis=0)
            upper_y = np.percentile(y_sims,q=97.5,axis=0)
            lower_y = np.percentile(y_sims,q=2.5,axis=0)
    
            # plots
            dates = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in datetimes]
            pred_dates = dates + [dates[-1] + dt.timedelta(days=i) for i in range(1,1 + n_predictions)]

            np.savetxt(region + os.path.sep + "y_obs.csv", y_obs, delimiter = ',')

            dates_csv = pd.DataFrame(pred_dates).to_csv(region + os.path.sep + 'dates.csv', header=False, index=False)
            # linear plot
            font_size = 12
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(dates, y_obs, 'x', color='k', label='reported data')
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1,15)))
            plt.title(region[0].upper() + region[1:].lower() + "'s daily infections", fontsize = font_size)
            plt.xlabel('Date', fontsize = font_size)
            plt.ylabel('New daily infections', fontsize = font_size)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # plot propagated uncertainty
            plt.plot(pred_dates, mean_y, color='g', lw=2, label='mean')
            plt.fill_between(pred_dates, lower_y, upper_y, color='darkseagreen', label='95% credible interval')
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
    parser.add_argument('--no_post_plot', '-npp', action='store_true', default=False, help='flag indicating not to plot posterior distributions')
    parser.add_argument('--smart_prior', '-sp', action='store_true', default=False, help='flag indicating to use olde posterior distributions as new prior distributions')

    args = parser.parse_args()

    folder = args.folder
    region = args.region.lower()
    load_trace = args.load_trace
    plot_post = not args.no_post_plot
    propagate = not args.no_propagate
    smart_prior = args.smart_prior

    assert not (smart_prior and load_trace), "choose either to load trace or to use it as the prior distributions"

    run(region, folder, smart_prior, load_trace, propagate, plot_post)

if __name__ == '__main__':
    main()
