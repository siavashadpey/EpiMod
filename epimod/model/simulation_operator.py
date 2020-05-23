
from abc import ABCMeta, abstractmethod
import copy

import numpy as np
import theano as theano

from epimod.solver.ode_solver.ode_solver import ODESolver
from epimod.eqn.equation import Equation

class CachedSimulation(metaclass=ABCMeta):
	def __init__(self, ode_solver):
		self._ode_solver = ode_solver
		self._eqn_parameters = None
		self._cached_output = None
		self._cached_doutput_dparams = None
		self._grad_flag = True
		self._ode_solver.set_output_gradient_flag(self._grad_flag)
		self._cached_ode_inputs = None
		self._cached_eqn_parameters = None

	def __call__(self, parameters):
		self._eqn_parameters = parameters
		if not self._is_cached():
			self._update_parameters()
			self._ode_solver.solve()
			if self._grad_flag:
				(t, self._cached_output, self._cached_doutput_dparams) = self._ode_solver.get_outputs()
			else:
				(t, self._cached_output) = self._ode_solver.get_outputs()
				self._cached_doutput_dparams = None
			self._cache_inputs_and_parameters()

		return (self._cached_output, self._cached_doutput_dparams)

	def set_gradient_flag(self, flag):
		self._grad_flag = flag
		self._ode_solver.set_output_gradient_flag(self._grad_flag)

	def _is_cached(self):
		if self._cached_ode_inputs is None or self._cached_eqn_parameters is None:
			return False

		inputs = (self._ode_solver.initial_time, self._ode_solver.final_time, self._ode_solver.n_steps)

		flag = inputs == self._cached_ode_inputs and np.array_equal(self._cached_eqn_parameters, self._eqn_parameters)
		return flag

	@abstractmethod
	def _update_parameters(self):
		pass

	def _cache_inputs_and_parameters(self):
		self._cached_ode_inputs = (self._ode_solver.initial_time, self._ode_solver.final_time, self._ode_solver.n_steps)
		self._cached_eqn_parameters = copy.deepcopy(self._eqn_parameters)

class ModelOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim
		self._grad_op = ModelGradOp(cached_sim)

	def make_node(self, parameters):
		inputs = theano.tensor.as_tensor_variable(parameters)
		outdim = 2 # n_outputs x n_time_array
		outputs = (theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(inputs.dtype),
                      broadcastable=[False] * outdim)())
		node = theano.Apply(op = self, inputs = [inputs], outputs = [outputs])
		return node

	def perform(self, node, inputs, outputs):
		x = inputs[0]
		out = outputs[0]
		out[0] = self._cached_sim(x)[0]

	def grad(self, inputs, output_grads):
		x = inputs[0]
		g = output_grads[0]
		grad = self._grad_op(x, g)
		return [grad] # brackets are needed. if not theano len() error is raised

class ModelGradOp(theano.Op):
	def __init__(self, cached_sim):
		self._cached_sim = cached_sim

	def make_node(self, parameters, dL_df):
		parameters = theano.tensor.as_tensor_variable(parameters)
		dL_df = theano.tensor.as_tensor_variable(dL_df)
		outdim = 1 # n_parameters (gradient of obj function is a 1D vector)
		outputs = (theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(dL_df.dtype),
                      broadcastable=[False] * outdim)())
		node = theano.Apply(op = self, inputs = [parameters, dL_df], outputs = [outputs])
		return node

	def perform(self, node, inputs, outputs):
		x = inputs[0]
		#print(x.shape[0])
		dL_df = inputs[1]
		df_dparams = self._cached_sim(x)[1]
		out = outputs[0]
		grad = dL_df @ df_dparams[0,:,0:].T
		#print(grad.shape)
		grad = grad[:,0:x.shape[0]] # comment out 
		#print(grad.shape)
		out[0] = grad[0]