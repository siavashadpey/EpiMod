from abc import ABCMeta, abstractmethod

class ODESolver(metaclass=ABCMeta):
	def __init__(self, ti, tf, n_steps = 1):
		self._initial_time = ti
		self._final_time = tf
		self._n_steps = n_steps
		self._time = -1

		self._equation = None
		self._is_output_grad_needed = False
		self._initial_state = None
		self._initial_dstate_dp = None

		self._state = None
		self._dstate_dp = None
		self._outputs = None
		self._doutputs_dp = None

		self._output_frequency = 100
		self._is_output_stored = False
		self._outputs_array = None
		self._doutputs_dp_array = None
		self._time_array = None

	@property
	def initial_time(self):
		return self._initial_time
	
	@initial_time.setter
	def initial_time(self, value):
		self._initial_time = value

	@property
	def final_time(self):
		return self._final_time
	
	@final_time.setter
	def final_time(self, value):
		self._final_time = value

	@property
	def n_steps(self):
		return self._n_steps
	
	@n_steps.setter
	def n_steps(self, value):
		self._n_steps = value

	def time(self):
		return n._time

	@property
	def equation(self):
		return self._equation

	@equation.setter
	def equation(self, eqn):
		self._equation = eqn
	
	def state(self):
		if not self._is_output_grad_needed:
			return self._state
		else:
			return (self._state, self._dstate_dp)


	def set_initial_condition(self, state, dstate_dp = None):
		self._initial_state = state
		if dstate_dp is not None:
			self._initial_dstate_dp = dstate_dp
			self._is_output_grad_needed = True
		else:
			self._is_output_grad_needed = False

	@property
	def output_frequency(self):
		return self._output_frequency

	@output_frequency.setter
	def output_frequency(self, freq):
		self._output_frequency = freq
	
	def set_output_storing_flag(self, flag):
		self._is_output_stored = flag

	def set_output_gradient_flag(self, flag):
		self._is_output_grad_needed = flag

	def get_outputs(self):
		if self._is_output_stored:
			if self._is_output_grad_needed:
				return (self._time_array, self._outputs_array, self._doutputs_dp_array)
			else:
				return (self._time_array, self._outputs_array)
		else:
			if self._is_output_grad_needed:
				return (self._outputs, self._doutputs_dp)
			else:
				return self._outputs

	@abstractmethod
	def solve(self):
		pass
