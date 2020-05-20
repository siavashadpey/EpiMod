import unittest
import numpy as np
from epimod.eqn.seir import Seir

class TestSeir(unittest.TestCase):

	def test_constructor(self):
		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		self.assertEqual(eqn.sigma, 3.3)
		self.assertEqual(eqn.gamma, 1)
		self.assertEqual(eqn.beta, 2.2)
		self.assertEqual(eqn.n_components(),4)
		self.assertEqual(eqn.n_parameters(),3)
		self.assertEqual(eqn.n_outputs(),1)

	def test_source(self):

		u = np.array([100., 90., 110., 220.])

		eqn = Seir(beta=2.2,gamma=1,sigma=3.3)
		s = eqn.source(u=u)

		s_act = np.array([-2.2*u[0]*u[2],
					 	   2.2*u[0]*u[2] - 3.3*u[1],
					 	   3.3*u[1] - 1*u[2],
					 	   1*u[2]])
		
		self.assertAlmostEqual(np.sum(s), 0.0, 1E-7)
		np.testing.assert_almost_equal(s, s_act, 1E-7)

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
			np.testing.assert_almost_equal(ds_du[:,i], (s_p1 - s_m1)/(2*epsi), 1E-7)

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
		np.testing.assert_almost_equal(ds_dp[:,0], (s_p1 - s_m1)/(2*epsi), 1E-7)

		# sigma gradient
		eqn = Seir(beta, sigma + epsi, gamma)
		s_p1 = eqn.source(u=u)
		eqn = Seir(beta, sigma - epsi, gamma)
		s_m1 = eqn.source(u=u)
		np.testing.assert_almost_equal(ds_dp[:,1], (s_p1 - s_m1)/(2*epsi), 1E-7)

		# gamma gradient
		eqn = Seir(beta, sigma, gamma + epsi)
		s_p1 = eqn.source(u=u)
		eqn = Seir(beta, sigma, gamma - epsi)
		s_m1 = eqn.source(u=u)
		np.testing.assert_almost_equal(ds_dp[:,2], (s_p1 - s_m1)/(2*epsi), 1E-7)

if __name__ == '__main__':
	unittest.main()