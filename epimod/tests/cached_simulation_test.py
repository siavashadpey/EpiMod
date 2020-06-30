import unittest

import numpy as np

from epimod.solver.ode_solver.rk_solver import RKSolver
from epimod.eqn.seir import Seir
from epimod.model.simulation_operator import CachedSimulation

class CachedSEIRSimulation(CachedSimulation):
    def _update_parameters(self):
        (beta, sigma, gamma, kappa) = np.array(self._eqn_parameters, dtype=np.float64)
        eqn = self._ode_solver.equation
        eqn.beta = beta
        eqn.sigma = sigma
        eqn.gamma = gamma
        eqn.kappa = kappa

class TestCachedSimulation(unittest.TestCase):
    def test_correct_caching(self):
        # setup equation
        eqn = Seir(population = 1)

        # setup ode solver
        ti = 0.
        tf = 2.
        n_steps = 3
        rk = RKSolver(ti, tf, n_steps)
        rk.output_frequency = 1
        rk.set_output_storing_flag(True)
        rk.equation = eqn
        u0 = np.array([100., 0., 10., 0.])
        du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
        rk.set_initial_condition(u0, du0_dp)
        rk.set_output_gradient_flag(True)

        # setup cached simulation object
        cached_sim = CachedSEIRSimulation(rk)

        params = np.array([2.3, 0.2, 1./3., 1./4.])

        (f, df) = cached_sim(params)

        f1 = np.copy(f)
        df1 = np.copy(df)

        params2 = np.array([2.32, 0.2, 1./3., 1./4.])
        (f2, df2) = cached_sim(params2)

        assert not np.allclose(f1, f2)
        assert not np.allclose(df1, df2)


        rk.final_time = 3
        (f3, df3) = cached_sim(params)

        assert not np.allclose(f1, f3)
        assert not np.allclose(df1, df3)    

    def test_correct_caching2(self):
        # setup equation
        eqn = Seir(population = 1)

        # setup ode solver
        ti = 0.
        tf = 2.
        n_steps = 3
        rk = RKSolver(ti, tf, n_steps)
        rk.output_frequency = 1
        rk.set_output_storing_flag(True)
        rk.equation = eqn
        u0 = np.array([100., 0., 10., 0.])
        du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
        rk.set_initial_condition(u0, du0_dp)
        rk.set_output_gradient_flag(True)

        # setup cached simulation object
        cached_sim = CachedSEIRSimulation(rk)

        params = np.array([2.3, 0.2, 1./3., 1./4.])

        (f, df) = cached_sim(params)

        f1 = np.copy(f)
        df1 = np.copy(df)

        f *= 0.
        df += 0.

        (f2, df2) = cached_sim(params)
        
        assert np.allclose(f1, f2)
        assert np.allclose(df1, df2)

    def test_correct_caching3(self):
        # setup equation
        eqn = Seir(population = 1)

        # setup ode solver
        ti = 0.
        tf = 2.
        n_steps = 3
        rk = RKSolver(ti, tf, n_steps)
        rk.output_frequency = 1
        rk.set_output_storing_flag(True)
        rk.equation = eqn
        u0 = np.array([100., 0., 10., 0.])
        rk.set_initial_condition(u0)

        # setup cached simulation object
        cached_sim = CachedSEIRSimulation(rk)
        cached_sim.set_gradient_flag(False)

        params = np.array([2.3, 0.2, 1./3., 1./4.])

        f = cached_sim(params)[0]

        f1 = np.copy(f)

        params2 = np.array([2.32, 0.2, 1./3., 1./4.])

        f2 = cached_sim(params2)[0]

        assert not np.allclose(f1, f2)  

if __name__ == '__main__':
    unittest.main()