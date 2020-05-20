import unittest
from epimod.eqn.seir import Seir

class TestSeir(unittest.TestCase):

	def test_constructor(self):
		eqn = Seir()
		print(eqn.beta)

unittest.main()