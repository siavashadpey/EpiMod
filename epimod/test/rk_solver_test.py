import unittest
import numpy as np
import math
from epimod.solver.ode_solver.rk_solver import RKSolver
from epimod.eqn.equation import Equation
from epimod.eqn.seir import Seir

class TestRKSolver(unittest.TestCase):
	def test_constructor(self):
		rk = RKSolver(0.5, 3, 2)
		self.assertEqual(rk.initial_time,0.5)
		self.assertEqual(rk.final_time,3)
		self.assertEqual(rk.n_steps,2)

	def test_interface(self):
		rk = RKSolver(0, 1)
		rk.initial_time = 0.5
		rk.final_time = 5
		rk.n_steps = 10
		self.assertEqual(rk.initial_time,0.5)
		self.assertEqual(rk.final_time,5)
		self.assertEqual(rk.n_steps,10)

		eqn = Seir()
		rk.equation = eqn
		self.assertEqual(rk.equation, eqn)

		rk.output_frequency = 20
		self.assertEqual(rk.output_frequency, 20)

	def test_rk4_solver(self):
		return
		theta = -1.2
		eqn = SimpleEquation(theta)
		u_exact = lambda t: eqn.solution(t)[0]

		ti = 0
		tf = 5*math.pi/3
		u0 = u_exact(ti)
		rk = RKSolver(ti, tf)
		rk.equation = eqn
		rk.set_initial_condition(u0)

		n = 4
		err = np.zeros((n,))
		dt = np.zeros((n,))
		n_steps = 100
		for i in range(n):
			n_steps *= (i+1)
			dt[i] = 1./n_steps
			rk.n_steps = n_steps
			rk.solve()
			err[i] = abs(rk.state() -  u_exact(tf))

		conv_rate = np.polyfit(np.log(dt), np.log(err), 1)[0]
		self.assertAlmostEqual(conv_rate, 4.0, 1)

	def test_rk4_gradient_computation(self):
		theta = -1.2
		eqn = SimpleEquation(theta)

		ti = 0
		tf = 5*math.pi/3
		(u0, du0_dp) = eqn.solution(ti)
		rk = RKSolver(ti, tf)
		rk.equation = eqn
		rk.set_initial_condition(u0, du0_dp)
		rk.set_output_gradient_flag(True)
		n = 4
		err = np.zeros((n,))
		dt = np.zeros((n,))
		n_steps = 100
		for i in range(n):
			n_steps *= (i+1)
			dt[i] = 1./n_steps
			rk.n_steps = n_steps
			rk.solve()
			err[i] = abs(rk.state()[1] -  eqn.solution(tf)[1])

		conv_rate = np.polyfit(np.log(dt), np.log(err), 1)[0]
		self.assertAlmostEqual(conv_rate, 4.0, 1)

	def test_rk4_gradient_computation2(self):
		n_pop = 7E6
		sigma = 1/5.2
		gamma = 1/2.28
		beta = 2.13*gamma
		eqn = Seir(beta, sigma, gamma)

		ti = 0
		tf = 218
		n_steps = 2*tf
		rk = RKSolver(ti, tf, n_steps)
		rk.equation = eqn

		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)
		rk.set_output_gradient_flag(True)

		rk.solve()
		(u, du_dp) = rk.state()

		rk.set_output_gradient_flag(False)

		epsi = 0.001

		# perturb beta0
		eqn.beta = beta + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.beta = beta - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,0] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.beta = beta

		# perturb sigma
		eqn.sigma = sigma + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.sigma = sigma - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,1] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.sigma = sigma

		# perturb gamma
		eqn.gamma = gamma + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.gamma = gamma - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,2] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.gamma = gamma

		# perturb kappa
		kappa = 1
		eqn.kappa = kappa + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.kappa = kappa - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,3] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.kappa = kappa

	def test_rk4_gradient_computation3(self):
		n_pop = 7E6
		sigma = 1/5.2
		gamma = 1/2.28
		beta = 2.13*gamma
		kappa = 1.2
		tau = 14
		eqn = Seir(beta, sigma, gamma, kappa, tau)

		ti = 0
		tf = 20
		n_steps = 2*tf
		rk = RKSolver(ti, tf, n_steps)
		rk.equation = eqn

		u0 = np.array([n_pop - 1, 0, 1, 0])
		u0 /= n_pop
		du0_dp = np.zeros((eqn.n_components(), eqn.n_parameters()))
		rk.set_initial_condition(u0, du0_dp)
		rk.set_output_gradient_flag(True)

		rk.solve()
		(u, du_dp) = rk.state()

		rk.set_output_gradient_flag(False)

		epsi = 0.001

		# perturb beta0
		eqn.beta = beta + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.beta = beta - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,0] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.beta = beta

		# perturb sigma
		eqn.sigma = sigma + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.sigma = sigma - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,1] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.sigma = sigma

		# perturb gamma
		eqn.gamma = gamma + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.gamma = gamma - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,2] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.gamma = gamma

		# perturb kappa
		eqn.kappa = kappa + epsi
		rk.solve()
		u_p1 = rk.state()

		eqn.kappa = kappa - epsi
		rk.solve()
		u_m1 = rk.state()
		diff = np.linalg.norm(du_dp[:,3] - (u_p1 - u_m1)/(2*epsi))/np.linalg.norm(u0)
		np.testing.assert_almost_equal(diff, 0, 5)
		# reset
		eqn.kappa = kappa


class SimpleEquation(Equation):
	def __init__(self, theta):
		super().__init__()
		self._theta = theta

	def solution(self, t):
		u = math.exp(self._theta**3*t)
		du_dp = u*3*self._theta**2*t
		return (np.array([u]), np.array([du_dp]))

	def source(self, t, u, is_grad_needed = False):
		r = self._theta**3*u
		if not is_grad_needed:
			return r

		dr_du = np.array([self._theta**3])
		dr_dp = 3*self._theta**2*u
		return (r, dr_du, dr_dp)

	def output(self, t, u, du_dp):
		pass

if __name__ == '__main__':
	unittest.main()
