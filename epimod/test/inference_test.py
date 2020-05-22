import math
import numpy as np
import matplotlib.pyplot as plt

import theano as theano
import theano.tensor as tt

import pymc3 as pm
import arviz as az

from epimod.solver.ode_solver.rk_solver import RKSolver
from epimod.eqn.equation import Equation
from epimod.eqn.seir import Seir

theano.config.exception_verbosity= 'high'

class CachedSimulation:
	def __init__(self, ode_solver):
		self._ode_solver = ode_solver
		self._cached_output = None
		self._cached_doutput_dparams = None
		self._cached_beta = None
		self._cached_sigma = None
		self._cached_gamma = None
		self._grad_flag = True

	def __call__(self, beta, sigma, gamma):
		#print(beta)
		#print(sigma)
		#print(gamma)
		#print("")
		if not self._is_cached(beta, sigma, gamma):
			#print(beta)
			#print(sigma)
			#print(gamma)
			self._ode_solver.equation.beta  = beta
			self._ode_solver.equation.sigma = sigma
			self._ode_solver.equation.gamma = gamma
			self._ode_solver.set_output_gradient_flag(self._grad_flag)
			self._ode_solver.solve()
			if self._grad_flag:
				(t, self._cached_output, self._cached_doutput_dparams) = self._ode_solver.get_outputs()
			else:
				(t, self._cached_output) = self._ode_solver.get_outputs()
				self._cached_doutput_dparams = None
			#(t, self._cached_output) = self._ode_solver.get_outputs()
			self._cached_beta = beta
			self._cached_sigma = sigma
			self._cached_gamma = gamma

		return (self._cached_output, self._cached_doutput_dparams)

	def set_gradient_flag(self, flag):
		self._grad_flag = flag

	def _is_cached(self, beta, sigma, gamma):
		return (self._cached_beta == beta  and
			   	self._cached_sigma == sigma and
			    self._cached_gamma == gamma)

class SimModelGradOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim

	def make_node(self, beta, sigma, gamma, dL_df):
		beta = tt.as_tensor_variable(beta)
		sigma = tt.as_tensor_variable(sigma)
		gamma = tt.as_tensor_variable(gamma)
		dL_df = tt.as_tensor_variable(dL_df)
		inputs = [beta, sigma, gamma, dL_df]
		outputs = [(theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(beta.dtype),
                      broadcastable=[False] * 1)())]
		node = theano.Apply(op = self, inputs = inputs, outputs = outputs)
		return node

	def perform(self, node, inputs, outputs):
		(beta, sigma, gamma, dL_df) = inputs
		df_dparams = self._cached_sim(beta, sigma, gamma)[1]
		x = dL_df[0].dot(df_dparams[:,0].T)
		y = dL_df[0].dot(df_dparams[:,1].T)
		z = dL_df[0].dot(df_dparams[:,2].T)
		outputs[0][0] = [x.item(), y.item(), z.item()]

class SimModelOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim
		self._grad_op = SimModelGradOp(cached_sim)

	def make_node(self, beta, sigma, gamma):
		beta = tt.as_tensor_variable(beta)
		sigma = tt.as_tensor_variable(sigma)
		gamma = tt.as_tensor_variable(gamma)
		inputs = [beta, sigma, gamma]
		outputs = [(theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(beta.dtype),
                      broadcastable=[False] * 2)())]
		node = theano.Apply(op = self, inputs = inputs, outputs = outputs)
		return node

	def perform(self, node, inputs, outputs):
		(beta, sigma, gamma) = inputs
		outputs[0][0] = self._cached_sim(beta, sigma, gamma)[0]

	def grad(self, inputs, output_grads):
		(beta, sigma, gamma) = inputs
		grad = self._grad_op(beta, sigma, gamma, output_grads)
		return [grad[0], grad[1], grad[2]]

from theano.tests import unittest_tools as utt
import unittest

class TestSimModelOp(utt.InferShapeTester):
	rng = np.random.RandomState(43)

	def setUp(self):
		super(TestSimModelOp, self).setUp()
		eqn = Seir()

		# set ode solver
		ti = 0
		tf = 20
		n_steps = tf
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)

		# set cached_sim object
		cached_sim = CachedSimulation(rk)
		cached_sim.set_gradient_flag(False)

		# set theano model op object
		self.op_class = SimModelOp(cached_sim)

	def test_perform(self):
		b = theano.tensor.dscalar('myvar0')
		s = theano.tensor.dscalar('myvar1')
		g = theano.tensor.dscalar('myvar2')
		f = theano.function([b, s, g], self.op_class(b, s, g))
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

	#def test_gradient(self):
	#	s_val = 1./5.2 
	#	g_val = 1./2.28
	#	b_val = 2.13*g_val
	#	utt.verify_grad(self.op_class, [b_val, s_val, g_val], n_tests=1, rng=TestSimModelOp.rng)

class TestSimModelGradOp(utt.InferShapeTester):
	rng = np.random.RandomState(43)

	def setUp(self):
		super(TestSimModelGradOp, self).setUp()
		eqn = Seir()

		# set ode solver
		ti = 0
		tf = 20
		n_steps = tf
		rk = RKSolver(ti, tf, n_steps)

		rk.output_frequency = 1
		rk.set_output_storing_flag(True)
		rk.equation = eqn

		n_pop = 7E6
		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)
		rk.set_output_gradient_flag(True)

		# set cached_sim object
		cached_sim = CachedSimulation(rk)

		# set theano model op object
		self.op_class = SimModelGradOp(cached_sim)

	def test_perform(self):
		b = theano.tensor.dscalar('myvar0')
		s = theano.tensor.dscalar('myvar1')
		g = theano.tensor.dscalar('myvar2')
		dL_df = theano.tensor.matrix()
		f = theano.function([b, s, g, dL_df], self.op_class(b, s, g, dL_df))
		s_val = 1./5.2 
		g_val = 1./2.28
		b_val = 2.13*g_val
		dL_df_val = np.random.rand(1,21)
		out = f(b_val, s_val, g_val, dL_df_val)
		out_act = np.array([3.899076278035202e-06, 1.0157708257250106e-05, -8.113407197470448e-06])
		assert np.allclose(out_act, out)

if __name__ == '__main__':
	unittest.main()