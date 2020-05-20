import unittest
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

	#TODO: test source computation (and gradient)

	#TODO: test output computation (and gradient)


if __name__ == '__main__':
	unittest.main()