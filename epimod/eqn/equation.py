from abc import ABCMeta, abstractmethod

class Equation(metaclass=ABCMeta):
    def __init__(self):
        pass

    def n_components(self):
        return self._n_components

    def n_parameters(self):
        return self._n_parameters

    def n_outputs(self):
        return self._n_outputs  
    
    @abstractmethod
    def source(self, t, u, is_grad_needed):
        pass

    @abstractmethod
    def output(self, t, u, du_dp):
        pass