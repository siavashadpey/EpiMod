import argparse 
import math
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from epimod.eqn.seir import Seir
from epimod.solver.ode_solver.rk_solver import RKSolver
import epimod.data.read_region_data as data_fetcher

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
            
            y = np.diff(outs[1], prepend=0)
            y[y<1.E-14] = 1.E-14
            return (self._time_array, y)
        
        raise Exception("output is not stored")

def run(region, folder):

    (t_obs, dates, y_obs, n_pop, shutdown_day, u0, _) = data_fetcher.read_region_data(folder, region)
    y_obs = y_obs.astype(np.float64)
    u0 = u0.astype(np.float64)

    # set eqn
    eqn = Seir()
    eqn.tau = shutdown_day
    eqn.population = n_pop
    eqn.beta = .7
    eqn.sigma = 0.8
    eqn.gamma = 0.2
    eqn.kappa = 0.25
    eqn.tint = 25
    
    # set ode solver
    ti = t_obs[0]
    tf = t_obs[-1]
    m = 2
    n_steps = m*(tf-ti)
    rk = RKSolverSeir(ti, tf, n_steps)
    rk.rk_type = "explicit_euler"
    rk.output_frequency = m
    rk.set_output_storing_flag(True)
    rk.equation = eqn
    rk.set_initial_condition(u0)
    rk.set_output_gradient_flag(False)

    rk.solve()
    (_, y_sim) = rk.get_outputs()   

    x_dates = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in dates]

    fig, ax = plt.subplots()
    ax.plot(x_dates, y_sim[0,:], color='b', lw=2)
    ax.plot(x_dates, y_obs, 'x', color='b')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    plt.yscale('log')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Runs SEIR simulation for the specified region.")
    parser.add_argument('--folder', '-f', default='./data/', help='path of data folder')
    parser.add_argument('--region', '-r', default='canada', help='region of interest')

    args = parser.parse_args()

    folder = args.folder
    region = args.region.lower()
    run(region, folder)

if __name__ == '__main__':
    main()