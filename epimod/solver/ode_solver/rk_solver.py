import math
import numpy as np
from epimod.solver.ode_solver.ode_solver import ODESolver

#TODO: add RKType (e.g. RK1, RK4, ...)

class RKSolver(ODESolver):
    def __init__(self, ti, tf, n_steps = 1, type='rk4'):
        super().__init__(ti, tf, n_steps)
        self._rk_types = np.array(["explicit_euler", "rk1", "rk4"])
        self.rk_type = type
        self._initialize_rk_coefficients()

    @property
    def rk_type(self):
        return self._rk_type
    
    @rk_type.setter
    def rk_type(self, type):
        assert type.lower() in self._rk_types, type +  " is an unsupported RK type."
        self._rk_type = type.lower()
        self._initialize_rk_coefficients()


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
                self._time_array = np.zeros((n))
                self._time_array[0] = self._time
                self._outputs_array[:,0] = self._outputs
                if self._is_output_grad_needed:
                    self._doutputs_dp_array = np.zeros((self._equation.n_outputs(), self._equation.n_parameters(), n))
                    self._doutputs_dp_array[:,:,0] = self._doutputs_dp

        dt = (self._final_time - self._initial_time)/self._n_steps
        for istep in range(1,self._n_steps+1):
            (self._state, self._dstate_dp) = self._one_rk_solve(resfun, self._time, dt, self._state, self._dstate_dp, self._is_output_grad_needed)
            self._time += dt 

            # compute output if needed
            if istep % self._output_frequency == 0:
                if is_output_computed:
                    if self._is_output_grad_needed:
                        (self._outputs, self._doutputs_dp) = self._equation.output(self._time, self._state, self._dstate_dp)
                    else:
                        self._outputs = self._equation.output(self._time, self._state)

                    if self._is_output_stored:
                        i = math.floor(istep/self._output_frequency)
                        self._time_array[i] = self._time
                        self._outputs_array[:,i] = self._outputs
                        if self._is_output_grad_needed:
                            self._doutputs_dp_array[:,:,i] = self._doutputs_dp

    def _one_rk_solve(self, resfun, t, dt, u, dudp, is_grad_needed = False):
        A = self._A
        b = self._b
        c = self._c
        n_stages = b.shape[0]

        assert (A[i, i] != 0 for i in range(n_stages)), "implicit RK methods are not currently supported"
        if not is_grad_needed:
            res = np.zeros((n_stages, u.shape[0]))
            for i_stage in range(n_stages):
                du_stage = np.zeros_like(u)
                for j in range(i_stage):
                    du_stage += dt*A[i_stage,j]*res[j]

                res[i_stage] = resfun(t + c[i_stage]*dt, u + du_stage)

            for j in range(n_stages):
                u += dt*b[j]*res[j] 

            return (u, None)

        res = np.zeros((n_stages, u.shape[0]))
        dresdp = np.zeros((n_stages, dudp.shape[0], dudp.shape[1]))
        for i_stage in range(n_stages):
            du_stage = np.zeros_like(u)
            dudp_stage = np.zeros_like(dudp)
            for j in range(i_stage):
                du_stage += dt*A[i_stage,j]*res[j]
                dudp_stage += dt*A[i_stage,j]*dresdp[j]

            res[i_stage], dresdu_stage, dresdp[i_stage] = resfun(t + c[i_stage]*dt, u + du_stage)
            dresdp[i_stage] += dresdu_stage@(dudp + dudp_stage)

        for j in range(n_stages):
            u += dt*b[j]*res[j]
            dudp += dt*b[j]*dresdp[j]

        return (u, dudp)

    def _initialize_rk_coefficients(self):
        if self._rk_type == "rk1" or self._rk_type == "explicit_euler":
            self._A = np.zeros((1, 1))
            self._b = np.array([1.])
            self._c = np.array([0.])
        elif self._rk_type == "rk4":
            self._A = np.zeros((4, 4))
            self._A[1,0] = 0.5
            self._A[2,1] = 0.5
            self._A[3,2] = 1.0
            self._b = np.array([1./6., 1./3., 1./3., 1./6.])
            self._c = np.array([0., 0.5, 0.5, 1.])

