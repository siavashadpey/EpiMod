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
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		eqn = Seir()
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)

		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)
		cached_sim.set_gradient_flag(True)

		# set theano model op object
		self.op_class = ModelOp(cached_sim)

	def test_perform(self):
		b = theano.tensor.dscalar('myvar0')
		s = theano.tensor.dscalar('myvar1')
		g = theano.tensor.dscalar('myvar2')
		f = theano.function([b, s, g], self.op_class((b, s, g)))
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		out = f(b_val, s_val, g_val)
		out_act = np.array([1.42857143e-07, 1.01439369e-07, 8.65162795e-08, 8.46869028e-08,
  							 8.97221828e-08, 9.87819861e-08, 1.10634869e-07, 1.24821306e-07,
  							 1.41261498e-07, 1.60072825e-07, 1.81486334e-07, 2.05810192e-07,
  							 2.33415645e-07, 2.64733988e-07, 3.00259217e-07, 3.40553892e-07,
  							 3.86257136e-07, 4.38094351e-07, 4.96888522e-07, 5.63573190e-07,
  							 6.39207233e-07])
		assert np.allclose(out_act, out)

	def test_grad(self):
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		rng = np.random.RandomState(42)
		theano.tensor.verify_grad(self.op_class, [(b_val, s_val, g_val)], rng=rng)

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
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		eqn = Seir()
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)

		# set cached_sim object
		cached_sim = CachedSEIRSimulation(rk)
		cached_sim.set_gradient_flag(True)

		# set theano model op object
		self.op_class = ModelGradOp(cached_sim)

	def test_perform(self):
		b = theano.tensor.dscalar('myvar0')
		s = theano.tensor.dscalar('myvar1')
		g = theano.tensor.dscalar('myvar2')
		dL_df = theano.tensor.matrix()
		f = theano.function([b, s, g, dL_df], self.op_class((b, s, g), dL_df))
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		dL_df_val = np.random.rand(1,21)
		out = f(b_val, s_val, g_val, dL_df_val)
		out_act = np.array([3.899076278035202e-06, 1.0157708257250106e-05, -8.113407197470448e-06])
		assert np.allclose(out_act, out)

class CachedSEIRSimulation2(CachedSimulation):

	def _update_parameters(self):
		(beta, sigma, gamma, kappa) = np.array(self._eqn_parameters, dtype=np.float64) #self._eqn_parameters
		eqn = self._ode_solver.equation
		eqn.beta = beta
		eqn.sigma = sigma
		eqn.gamma = gamma
		eqn.kappa = kappa

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
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		eqn = Seir()
		eqn.tau = 14
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)

		# set cached_sim object
		cached_sim = CachedSEIRSimulation2(rk)
		cached_sim.set_gradient_flag(True)

		# set theano model op object
		self.op_class = ModelOp(cached_sim)

	def test_perform(self):
		b = theano.tensor.dscalar('myvar0')
		s = theano.tensor.dscalar('myvar1')
		g = theano.tensor.dscalar('myvar2')
		k = theano.tensor.dscalar('myvar3')
		f = theano.function([b, s, g, k], self.op_class((b, s, g, k)))
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		k_val = 1.1
		out = f(b_val, s_val, g_val, k_val)
		out_act = np.array([1.42857143e-07, 1.01439369e-07, 8.65162795e-08, 8.46869028e-08,
  							 8.97221828e-08, 9.87819861e-08, 1.10634869e-07, 1.24821306e-07,
  							 1.41261498e-07, 1.60072825e-07, 1.81486334e-07, 2.05810192e-07,
  							 2.33415645e-07, 2.64733988e-07, 3.00259217e-07, 3.33728312e-07,
  							 3.45812003e-07, 3.35998582e-07, 3.11617490e-07, 2.79821511e-07,
  							 2.45682946e-07])
		assert np.allclose(out_act, out)

	def test_grad(self):
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		k_val = 1.1
		rng = np.random.RandomState(42)
		theano.tensor.verify_grad(self.op_class, [(b_val, s_val, g_val, k_val)], rng=rng)

if __name__ == '__main__':
	unittest.main()