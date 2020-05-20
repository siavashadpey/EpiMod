import unittest
import numpy as np
from epimod.eqn.seir import Seir

decimal = 7
class TestSeir(unittest.TestCase):

	def test_constructor(self):
		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		self.assertEqual(eqn.sigma, 3.3)
		self.assertEqual(eqn.gamma, 1)
		self.assertEqual(eqn.beta, 2.2)
		self.assertEqual(eqn.n_components(),4)
		self.assertEqual(eqn.n_parameters(),3)
		self.assertEqual(eqn.n_outputs(),1)

	def test_interface(self):
		eqn = Seir()
		eqn.beta = 0.5
		eqn.sigma = 5.2
		eqn.gamma = 13
		self.assertEqual(eqn.sigma, 5.2)
		self.assertEqual(eqn.gamma, 13)
		self.assertEqual(eqn.beta, 0.5)

	def test_source(self):

		u = np.array([100., 90., 110., 220.])

		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		s = eqn.source(u=u)

		s_act = np.array([-2.2*u[0]*u[2],
					 	   2.2*u[0]*u[2] - 3.3*u[1],
					 	   3.3*u[1] - 1*u[2],
					 	   1*u[2]])
		
		self.assertAlmostEqual(np.sum(s), 0.0, decimal)
		np.testing.assert_almost_equal(s, s_act, decimal)

	def test_source_state_grad(self):
		u = np.array([100., 90., 110., 220.])

		beta = 2.2
		sigma = 3.3
		gamma = 4.13
		eqn = Seir(beta, sigma, gamma)
		(s, ds_du, ds_dp) = eqn.source(u=u, is_grad_needed = True)

		epsi = 1E-1

		for i in range(eqn.n_components()):
			u_p1 = np.copy(u)
			u_p1[i] += epsi
			s_p1 = eqn.source(u = u_p1)

			u_m1 = np.copy(u)
			u_m1[i] -= epsi
			s_m1 = eqn.source(u = u_m1)
			np.testing.assert_almost_equal(ds_du[:,i], (s_p1 - s_m1)/(2*epsi), decimal)

	def test_source_param_grad(self):
		u = np.array([100., 90., 110., 220.])

		beta = 2.2
		sigma = 3.3
		gamma = 4.13

		eqn = Seir(beta, sigma, gamma)
		(s, ds_du, ds_dp) = eqn.source(u=u, is_grad_needed = True)

		epsi = 1E-3

		# beta gradient
		eqn = Seir(beta + epsi, sigma, gamma)
		s_p1 = eqn.source(u=u)
		eqn = Seir(beta - epsi, sigma, gamma)
		s_m1 = eqn.source(u=u)
		np.testing.assert_almost_equal(ds_dp[:,0], (s_p1 - s_m1)/(2*epsi), decimal)

		# sigma gradient
		eqn = Seir(beta, sigma + epsi, gamma)
		s_p1 = eqn.source(u=u)
		eqn = Seir(beta, sigma - epsi, gamma)
		s_m1 = eqn.source(u=u)
		np.testing.assert_almost_equal(ds_dp[:,1], (s_p1 - s_m1)/(2*epsi), decimal)

		# gamma gradient
		eqn = Seir(beta, sigma, gamma + epsi)
		s_p1 = eqn.source(u=u)
		eqn = Seir(beta, sigma, gamma - epsi)
		s_m1 = eqn.source(u=u)
		np.testing.assert_almost_equal(ds_dp[:,2], (s_p1 - s_m1)/(2*epsi), decimal)

	def test_output(self):
		u = np.array([100., 90., 110., 220.])

		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		s = eqn.source(u=u)
		f = eqn.output(u=u)

		self.assertAlmostEqual(f, u[2], decimal)

	def test_output_param_grad(self):
		u = np.array([100., 90., 110., 220.])

		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		s = eqn.source(u=u)
		du_dp = np.random.random((eqn.n_components(), eqn.n_parameters()))
		(f, df_dp) = eqn.output(u=u, du_dp = du_dp)

		np.testing.assert_almost_equal(df_dp, du_dp[2,:], decimal)


if __name__ == '__main__':
	unittest.main()