import unittest
from theano.tests import unittest_tools as utt

import numpy as np
import theano as theano

from epimod.model.simulation_operator import CachedSimulation
from epimod.model.simulation_operator import ModelOp
from epimod.model.simulation_operator import ModelGradOp
from epimod.eqn.seir import Seir
from epimod.solver.ode_solver.rk_solver import RKSolver

class CachedSEIRSimulation(CachedSimulation):

    def _update_parameters(self):
        (beta, sigma, gamma) = np.array(self._eqn_parameters, dtype=np.float64) #self._eqn_parameters
        eqn = self._ode_solver.equation
        eqn.beta = beta
        eqn.sigma = sigma
        eqn.gamma = gamma

class TestModelOp(utt.InferShapeTester):
    rng = np.random.RandomState(43)

    def setUp(self):
        super(TestModelOp, self).setUp()
        self.setUpModel()

    def setUpModel(self):
        # set ode solver
        ti = 0
        tf = 20
        n_steps = tf
        self.rk = RKSolver(ti, tf, n_steps)

        self.rk.output_frequency = 1
        self.rk.set_output_storing_flag(True)
        eqn = Seir()
        self.rk.equation = eqn

        n_pop = 7E6
        u0 = np.array([n_pop - 1, 0, 1, 0])
        u0 /= n_pop
        du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
        self.rk.set_initial_condition(u0, du0_dp)

        # set cached_sim object
        cached_sim = CachedSEIRSimulation(self.rk)
        cached_sim.set_gradient_flag(True)

        # set theano model op object
        self.op_class = ModelOp(cached_sim)

    #@unittest.skip("changed output of SEIR eq")
    def test_perform(self):
        b = theano.tensor.dscalar('myvar0')
        s = theano.tensor.dscalar('myvar1')
        g = theano.tensor.dscalar('myvar2')
        f = theano.function([b, s, g], self.op_class((b, s, g)))
        s_val = 1./5.2 
        g_val = 1./2.28
        b_val = 2.13*g_val
        out = f(b_val, s_val, g_val)
        self.rk.equation.beta = b_val
        self.rk.equation.sigma = s_val
        self.rk.equation.gamma = g_val
        self.rk.solve()
        (_,out_act, _) = self.rk.get_outputs()
        assert np.allclose(out_act, out)

    def test_grad(self):
        s_val = 1./5.2 
        g_val = 1./2.28
        b_val = 2.13*g_val
        rng = np.random.RandomState(42)
        theano.tensor.verify_grad(self.op_class, [(b_val, s_val, g_val)], rng=rng)

class CachedSEIRSimulation2(CachedSimulation):

    def _update_parameters(self):
        (beta, sigma, gamma, kappa, tint) = np.array(self._eqn_parameters, dtype=np.float64) #self._eqn_parameters
        eqn = self._ode_solver.equation
        eqn.beta = beta
        eqn.sigma = sigma
        eqn.gamma = gamma
        eqn.kappa = kappa
        eqn.tint = tint

class TestModelOp2(utt.InferShapeTester):
    rng = np.random.RandomState(43)

    def setUp(self):
        super(TestModelOp2, self).setUp()
        self.setUpModel()

    def setUpModel(self):
        # set ode solver
        ti = 0
        tf = 20
        n_steps = tf
        self.rk = RKSolver(ti, tf, n_steps)

        self.rk.output_frequency = 1
        self.rk.set_output_storing_flag(True)
        eqn = Seir()
        eqn.tau = 14
        self.rk.equation = eqn

        n_pop = 7E6
        u0 = np.array([n_pop - 1, 0, 1, 0])
        u0 /= n_pop
        du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
        self.rk.set_initial_condition(u0, du0_dp)

        # set cached_sim object
        cached_sim = CachedSEIRSimulation2(self.rk)
        cached_sim.set_gradient_flag(True)

        # set theano model op object
        self.op_class = ModelOp(cached_sim)

    def test_perform(self):
        b = theano.tensor.dscalar('myvar0')
        s = theano.tensor.dscalar('myvar1')
        g = theano.tensor.dscalar('myvar2')
        k = theano.tensor.dscalar('myvar3')
        t = theano.tensor.dscalar('myvar4')
        f = theano.function([b, s, g, k, t], self.op_class((b, s, g, k, t)))
        s_val = 1./5.2 
        g_val = 1./2.28
        b_val = 2.13*g_val
        k_val = 1.1
        t_val = 10
        out = f(b_val, s_val, g_val, k_val, t_val)
        self.rk.equation.beta = b_val
        self.rk.equation.sigma = s_val
        self.rk.equation.gamma = g_val
        self.rk.equation.kappa = k_val
        self.rk.equation.tint = t_val
        self.rk.solve()
        (_, out_act, _) = self.rk.get_outputs()
        assert np.allclose(out_act, out)

    def test_grad(self):
        s_val = 1./5.2 
        g_val = 1./2.28
        b_val = 2.13*g_val
        k_val = 1.1
        t_int = 10
        rng = np.random.RandomState(42)
        theano.tensor.verify_grad(self.op_class, [(b_val, s_val, g_val, k_val, t_int)], rng=rng)

class TestModelGradOp(utt.InferShapeTester):
    rng = np.random.RandomState(43)

    def setUp(self):
        super(TestModelGradOp, self).setUp()
        self.setUpModel()

    def setUpModel(self):
        # set ode solver
        ti = 0
        tf = 20
        n_steps = tf
        self.rk = RKSolver(ti, tf, n_steps)

        self.rk.output_frequency = 1
        self.rk.set_output_storing_flag(True)
        eqn = Seir()
        eqn.tau = 5
        self.rk.equation = eqn

        n_pop = 7E6
        u0 = np.array([n_pop - 1, 0, 1, 0])
        u0 /= n_pop
        du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
        self.rk.set_initial_condition(u0, du0_dp)

        # set cached_sim object
        cached_sim = CachedSEIRSimulation2(self.rk)
        cached_sim.set_gradient_flag(True)

        # set theano model op object
        self.op_class = ModelGradOp(cached_sim)

    def test_perform(self):
        b = theano.tensor.dscalar('myvar0')
        s = theano.tensor.dscalar('myvar1')
        g = theano.tensor.dscalar('myvar2')
        k = theano.tensor.dscalar('myvar3')
        t = theano.tensor.dscalar('myvar4')
        dL_df = theano.tensor.matrix()
        f = theano.function([b, s, g, k, t, dL_df], self.op_class((b, s, g, k, t), dL_df))
        s_val = 1./5.2 
        g_val = 1./2.28
        b_val = 2.13*g_val
        k_val = 1.1
        t_val = 10
        dL_df_val = np.random.rand(1,21)
        out = f(b_val, s_val, g_val, k_val, t_val, dL_df_val)
        self.rk.equation.beta = b_val
        self.rk.equation.sigma = s_val
        self.rk.equation.gamma = g_val
        self.rk.equation.kappa = k_val
        self.rk.equation.tint = t_val
        self.rk.solve()
        (_, _, df_dp) = self.rk.get_outputs()
        out_act = df_dp[0,:,:] @ dL_df_val.T
        out_act = np.reshape(out_act, (self.rk.equation.n_parameters(),))
        assert np.allclose(out_act, out)

if __name__ == '__main__':
    unittest.main()