import time
import os 

import argparse 
import math

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from epimod.eqn.seir import Seir
from epimod.solver.ode_solver.rk_solver import RKSolver
import epimod.data.read_region_data as data_fetcher

import multiprocessing
from multiprocessing import Pool

print(multiprocessing.cpu_count())

def set_and_sim(rk, beta):#, sigma):
    print(os.getpid(), beta)
    #print(1, beta); sys.stdout.flush()
    rk.equation.beta = beta
        #rk.equation.sigma = sigma
    rk.solve()
    (_, y_sim) = rk.get_outputs()
    print(os.getpid(), rk.equation.beta)
    #   print(3, rk.equation.beta); sys.stdout.flush()
    #   #print(4, y_sim)        
    return y_sim

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

def run(region, folder):

    (t_obs, dates, y_obs, n_pop, shutdown_day, u0, _) = data_fetcher.read_region_data(folder, region)
    y_obs = y_obs.astype(np.float64)
    u0 = u0.astype(np.float64)

    # set eqn
    eqn = Seir()
    eqn.tau = shutdown_day
    eqn.beta = 1.5E-8
    eqn.sigma = 0.45
    eqn.gamma = 0.25
    eqn.kappa = 0.36
    eqn.tint = 40.
    
    # set ode solver
    ti = t_obs[0]
    tf = t_obs[-1]
    m = 2000
    n_steps = m*(tf-ti)
    rk = RKSolverSeir(ti, tf, n_steps)
    rk.rk_type = "explicit_euler"
    rk.output_frequency = m
    rk.set_output_storing_flag(True)
    rk.equation = eqn
    rk.set_initial_condition(u0)
    rk.set_output_gradient_flag(False)

    betas = np.array([1.4E-8, 1.9E-8])
    sigmas = np.array([0.43, 0.47])
    y_sims = np.zeros((3, eqn.n_outputs(), int(rk.n_steps/rk.output_frequency) + 1))
    import concurrent.futures
    import copy

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(set_and_sim(copy.deepcopy(rk), betas[i])) for i in range(betas.shape[0])]
        #executor.submit(set_and_sim(copy.deepcopy(rk), betas[0]))
        #executor.submit(set_and_sim(copy.deepcopy(rk), betas[1]))
        concurrent.futures.wait(futures)

    print(time.time() - start)
    #print("hello", y_sims)
    #plt.plot(t_obs, y_sims[0, 0,:], color='b', lw=2)
    #plt.plot(t_obs, y_sims[1, 0,:], color='b', lw=2)
    #plt.plot(t_obs, y_sims[2, 0,:], color='b', lw=2)
    plt.plot(t_obs, y_obs, 'x', color='b')

    plt.savefig("sim.pdf")
    #plt.plot(t_obs, y_sim[0,:], color='b', lw=2)
    #plt.plot(t_obs, y_sim[1,:], color='b', lw=2)

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
