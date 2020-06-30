import math

import numpy as np

from epimod.eqn.equation import Equation

class Seir(Equation):
    def __init__(self, beta=0, sigma=0, gamma=0, kappa = 1, tau = math.inf, tint = 5, population = 1E6):
        super().__init__()
        self._n_components = 4
        self._n_parameters = 5
        self._n_outputs = 1
        self._beta0 = beta
        self._sigma = sigma
        self._gamma = gamma
        self._kappa = kappa
        self._tau = tau
        self._tint = tint
        self._population = population

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        self._population = value

    @property
    def beta(self):
        #TODO: output the actual time-varying beta
        return self._beta0

    @beta.setter
    def beta(self, value):
        self._beta0 = value # beta0: transmissibility rate

    @property
    def sigma(self):
        return self._sigma # 1/sigma: latent period (period between being exposed and bein infectious)
        # incubation period is period between being exposed and showing symptoms
        # however, a person can be infectious before showing symptoms
        # i.e. latent period could be shorter than incubation period

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def gamma(self):
        return self._gamma #1/gamma: infectious period 
        # infectious period + latent/incubation period = serial period (interval between person A being infected and infecting person B)

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def tau(self):
        return self._tau # day when social distancing measures are introduced
    
    @tau.setter
    def tau(self, value):
        self._tau = value
    
    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value # represents social distancing measures effectiveness

    @property
    def tint(self):
        return self._tint

    @tint.setter
    def tint(self, value):
        self._tint = value 

    def source(self, t = 0, u = 0, is_grad_needed = False):

        assert u.shape == (self._n_components,), "u is not a vector of size n_components"
        
        S = u[0]
        E = u[1]
        I = u[2]
        # R = u[3] # Not used

        (b, db0, dkappa, dtint) = Seir._compute_beta(self._beta0, self._kappa, self._tau, self._tint, t)
        s = self._sigma
        g = self._gamma
        N = self._population

        #print(b)
        f = np.array([
            -b*S*I/N,
             b*S*I/N - s*E,
             s*E - g*I, 
             g*I])

        if not is_grad_needed:
            return f

        df_du = np.array([[-b*I/N,  0, -b*S/N, 0],
                          [ b*I/N, -s,  b*S/N, 0],
                          [   0,  s,   -g, 0],
                          [   0,  0,    g, 0]])

        # rows: df_dbeta0, df_dsigma, df_dgamma, df_dkappa, df_dtint
        df_dp = np.array([[-db0*S*I/N,  0,  0, -dkappa*S*I/N, -dtint*S*I/N],
                          [ db0*S*I/N, -E,  0,  dkappa*S*I/N,  dtint*S*I/N],
                          [       0,  E, -I,           0,           0],
                          [       0,  0,  I,           0,           0]])

        return (f, df_du, df_dp)

    def output(self, t = 0, u = 0, du_dp = None):

        assert u.shape == (self._n_components,) , "u is not a vector of size n_components"

        if du_dp is None:
            return np.sum(u[1:])

        assert du_dp.shape == (self._n_components, self._n_parameters), "du_dp is not of size n_components x n_parameters"
        return (np.sum(u[1:]), np.sum(du_dp[1:,:],axis=0))

    @staticmethod
    def _compute_beta(theta0, theta1, tau, dt, t):
        if t <= tau:
            return (theta0, 1, 0, 0)
        elif t >= tau + dt:
            return (theta1*theta0, theta1, theta0, 0)
        elif t > tau and t < tau + dt:
            ratio = (t-tau)/dt
            return (theta0 + (theta1 - 1)*theta0*ratio, 1 + (theta1 - 1)*ratio, theta0*ratio, (theta1 - 1)*theta0*(-ratio)/dt)
        raise

