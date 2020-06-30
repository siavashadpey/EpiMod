import unittest
import math

import numpy as np

from epimod.eqn.seir import Seir

decimal = 6
class TestSeir(unittest.TestCase):

    def test_constructor(self):
        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, kappa = 5.1, tau = 1, population = 1)
        self.assertEqual(eqn.sigma, 3.3)
        self.assertEqual(eqn.gamma, 1)
        self.assertEqual(eqn.beta, 2.2)
        self.assertEqual(eqn.kappa, 5.1)
        self.assertEqual(eqn.tau, 1)
        self.assertEqual(eqn.n_components(),4)
        self.assertEqual(eqn.n_parameters(),5)
        self.assertEqual(eqn.n_outputs(),1)

    def test_interface(self):
        eqn = Seir()
        eqn.sigma = 5.2
        eqn.gamma = 13
        eqn.beta = 0.5
        eqn.tau = 2
        eqn.kappa = 1
        self.assertEqual(eqn.sigma, 5.2)
        self.assertEqual(eqn.gamma, 13)
        self.assertEqual(eqn.beta, 0.5)
        self.assertEqual(eqn.tau, 2)
        self.assertEqual(eqn.kappa, 1)

    def test_source(self):

        u = np.array([100., 90., 110., 220.])

        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, population = 1)
        s = eqn.source(u=u)

        s_act = np.array([-2.2*u[0]*u[2],
                           2.2*u[0]*u[2] - 3.3*u[1],
                           3.3*u[1] - 1*u[2],
                           1*u[2]])
        
        self.assertAlmostEqual(np.sum(s), 0.0, decimal)
        np.testing.assert_almost_equal(s, s_act, decimal)

    def test_source2(self):

        u = np.array([100., 90., 110., 220.])

        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, tau = 1, kappa = .3, population = 1)
        s = eqn.source(t=eqn.tau+7, u=u)

        beta = 2.2*0.3
        s_act = np.array([-beta*u[0]*u[2],
                           beta*u[0]*u[2] - 3.3*u[1],
                           3.3*u[1] - 1*u[2],
                           1*u[2]])
        
        self.assertAlmostEqual(np.sum(s), 0.0, decimal)
        np.testing.assert_almost_equal(s, s_act, decimal)

    def test_source3(self):

        u = np.array([100., 90., 110., 220.])

        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, tau = 1, kappa = .3, population = 1)
        s = eqn.source(t=eqn.tau+3, u=u)

        beta = 2.2 - 2.2*(1. - 0.3)/5*3
        s_act = np.array([-beta*u[0]*u[2],
                           beta*u[0]*u[2] - 3.3*u[1],
                           3.3*u[1] - 1*u[2],
                           1*u[2]])
        
        self.assertAlmostEqual(np.sum(s), 0.0, decimal)
        np.testing.assert_almost_equal(s, s_act, decimal)

    def test_source_state_grad(self):
        u = np.array([100., 90., 110., 220.])

        beta = 2.2
        sigma = 3.3
        gamma = 4.13
        eqn = Seir(beta, sigma, gamma, population = 5)
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

        beta0 = 2.2
        sigma = 3.3
        gamma = 4.13

        eqn = Seir(beta0, sigma, gamma, population = 5)
        (s, ds_du, ds_dp) = eqn.source(u=u, is_grad_needed = True)

        epsi = 1E-3

        # beta0 gradient
        eqn = Seir(beta0 + epsi, sigma, gamma, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0 - epsi, sigma, gamma, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,0], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # sigma gradient
        eqn = Seir(beta0, sigma + epsi, gamma, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0, sigma - epsi, gamma, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,1], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # gamma gradient
        eqn = Seir(beta0, sigma, gamma + epsi, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0, sigma, gamma - epsi, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,2], (s_p1 - s_m1)/(2*epsi), decimal)

        # kappa gradient
        kappa = 1.2
        eqn = Seir(beta0, sigma, gamma, kappa + epsi, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0, sigma, gamma, kappa - epsi, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,3], (s_p1 - s_m1)/(2*epsi), decimal)

        # tint gradient
        tint = eqn.tint
        eqn = Seir(beta0, sigma, gamma, kappa, tint=tint+epsi, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0, sigma, gamma, kappa, tint=tint-epsi, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,4], (s_p1 - s_m1)/(2*epsi), decimal)

    def test_source_param_grad2(self):
        u = np.array([100., 90., 110., 220.])

        beta0 = 2.2
        sigma = 3.3
        gamma = 4.13
        kappa = 1.2
        tau = 1
        t = tau + 7

        eqn = Seir(beta0, sigma, gamma, kappa, tau, population = 5)
        
        (s, ds_du, ds_dp) = eqn.source(t=t, u=u, is_grad_needed = True)

        epsi = 1E-5

        # beta0 gradient
        eqn = Seir(beta0 + epsi, sigma, gamma, kappa, tau, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0 - epsi, sigma, gamma, kappa, tau, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,0], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # sigma gradient
        eqn = Seir(beta0, sigma + epsi, gamma, kappa, tau, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma - epsi, gamma, kappa, tau, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,1], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # gamma gradient
        eqn = Seir(beta0, sigma, gamma + epsi, kappa, tau, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma, gamma - epsi, kappa, tau, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,2], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # kappa gradient
        eqn = Seir(beta0, sigma, gamma, kappa + epsi, tau, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma, gamma, kappa - epsi, tau, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,3], (s_p1 - s_m1)/(2*epsi), decimal)

        # tint gradient
        tint = eqn.tint
        eqn = Seir(beta0, sigma, gamma, kappa, tint=tint+epsi, population = 5)
        s_p1 = eqn.source(u=u)
        eqn = Seir(beta0, sigma, gamma, kappa, tint=tint-epsi, population = 5)
        s_m1 = eqn.source(u=u)
        np.testing.assert_almost_equal(ds_dp[:,4], (s_p1 - s_m1)/(2*epsi), decimal)

    def test_source_param_grad3(self):
        u = np.array([100., 90., 110., 220.])
        beta0 = 2.2
        sigma = 3.3
        gamma = 4.13
        kappa = 1.2
        tau = 1
        t = tau + 3
        tint = 5

        eqn = Seir(beta0, sigma, gamma, kappa, tau, tint, population = 5)
        
        (s, ds_du, ds_dp) = eqn.source(t=t, u=u, is_grad_needed = True)

        epsi = 1E-5

        # beta0 gradient
        eqn = Seir(beta0 + epsi, sigma, gamma, kappa, tau, tint, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0 - epsi, sigma, gamma, kappa, tau, tint, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,0], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # sigma gradient
        eqn = Seir(beta0, sigma + epsi, gamma, kappa, tau, tint, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma - epsi, gamma, kappa, tau, tint, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,1], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # gamma gradient
        eqn = Seir(beta0, sigma, gamma + epsi, kappa, tau, tint, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma, gamma - epsi, kappa, tau, tint, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,2], (s_p1 - s_m1)/(2*epsi), decimal)
        
        # kappa gradient
        eqn = Seir(beta0, sigma, gamma, kappa + epsi, tau, tint, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma, gamma, kappa - epsi, tau, tint, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,3], (s_p1 - s_m1)/(2*epsi), decimal)

        # tint gradient
        eqn = Seir(beta0, sigma, gamma, kappa, tau, tint + epsi, population = 5)
        s_p1 = eqn.source(t=t, u=u)
        eqn = Seir(beta0, sigma, gamma, kappa, tau, tint - epsi, population = 5)
        s_m1 = eqn.source(t=t, u=u)
        np.testing.assert_almost_equal(ds_dp[:,4], (s_p1 - s_m1)/(2*epsi), decimal)

    def test_output(self):
        u = np.array([100., 90., 110., 220.])

        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, population = 5)
        s = eqn.source(u=u)
        f = eqn.output(u=u)

        self.assertAlmostEqual(f, u[1]+u[2]+u[3], decimal)

    def test_output_param_grad(self):
        u = np.array([100., 90., 110., 220.])

        eqn = Seir(beta=2.2,gamma=1,sigma=3.3, population = 5)
        s = eqn.source(u=u)
        du_dp = np.random.random((eqn.n_components(), eqn.n_parameters()))
        (f, df_dp) = eqn.output(u=u, du_dp = du_dp)

        np.testing.assert_almost_equal(df_dp, du_dp[1,:] + du_dp[2,:] + du_dp[3,:], decimal)


if __name__ == '__main__':
    unittest.main()