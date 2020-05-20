import math
import numpy as np
from epimod.solver.ode_solver.ode_solver import ODESolver

#TODO: add RKType (e.g. RK1, RK4, ...)

class RKSolver(ODESolver):
	def __init__(self, ti, tf, n_steps = 1):
		super().__init__(ti, tf, n_steps)

	def solve(self):
		is_output_computed = self._is_output_stored

		# lambda functions for residual computation
		resfun = lambda t, state: self._equation.source(t, state, self._is_output_grad_needed)

		# initial condition
		self._state = np.copy(self._initial_state)
		self._dstate_dp = np.copy(self._initial_dstate_dp)
		self._time = self._initial_time

		# initial output
		if is_output_computed:
			if self._is_output_grad_needed:
				(self._outputs, self._doutputs_dp) = self._equation.output(self._time, self._state, self._dstate_dp)
			else:
				self._outputs = self._equation.output(self._time, self._state)

			if self._is_output_stored:
				n = math.floor(self._n_steps/self._output_frequency) + 1
				self._outputs_array = np.zeros((self._equation.n_outputs(), n))
				self._time_array = np.zeros((n,))
				self._outputs_array[:,0] = self._outputs
				self._time_array[0] = self._time
				if self._is_output_grad_needed:
					self._doutputs_dp_array = np.zeros((self._equation.n_outputs(), self._equation.n_parameters(), n))
					self._doutputs_dp_array[:,:,0] = self._doutputs_dp

		dt = (self._final_time - self._initial_time)/self._n_steps
		for istep in range(1,self._n_steps+1):
			(self._state, self._dstate_dp) = RKSolver._one_rk_solve(resfun, self._time, dt, self._state, self._dstate_dp, self._is_output_grad_needed)
			self._time += dt 

			# compute output if needed
			if istep % self._output_frequency == 0:
				if is_output_computed:
					if self._is_output_grad_needed:
						(self._outputs, self._doutputs_dp) = self._equation.output(self._time, self._state, self._dstate_dp)
					else:
						self._outputs = self._equation.output(self._time, self._state)

					if self._is_output_stored:
						i = math.floor(istep/self._output_frequency) + 1
						self._outputs_array[:,i] = self._outputs
						self._time_array[i] = self._time
						if self._is_output_grad_needed:
							self._doutputs_dp_array[:,:,i] = self._doutputs_dp

	@staticmethod
	def _one_rk_solve(resfun, t, dt, u, dudp, is_grad_needed = False):
		if not is_grad_needed:
			k1 = dt*resfun(t, u)
			k2 = dt*resfun(t + 0.5*dt, u + 0.5*k1)
			k3 = dt*resfun(t + 0.5*dt, u + 0.5*k2)
			k4 = dt*resfun(t + dt, u + k3)
			u += 1/6*(k1 + 2*k2 + 2*k3 + k4)
			return (u, None)
		
		(r1, dr1du, dr1dp) = resfun(t, u)
		k1 = dt*r1
		(r2, dr2du, dr2dp) = resfun(t + 0.5*dt, u + 0.5*k1)
		k2 = dt*r2
		(r3, dr3du, dr3dp) = resfun(t + 0.5*dt, u + 0.5*k2)
		k3 = dt*r3
		(r4, dr4du, dr4dp) = resfun(t + dt, u + k3)
		k4 = dt*r4
		u += 1/6*(k1 + 2*k2 + 2*k3 + k4)
		
		# gradient computation
		dk1dp = dt*(dr1dp + dr1du@dudp)
		dk2dp = dt*(dr2dp + dr2du@(dudp + 0.5*dk1dp))
		dk3dp = dt*(dr3dp + dr3du@(dudp + 0.5*dk2dp))
		dk4dp = dt*(dr4dp + dr4du@(dudp + dk3dp))
		dudp += 1/6*(dk1dp + 2*dk2dp + 2*dk3dp + dk4dp)

		return (u, dudp)
