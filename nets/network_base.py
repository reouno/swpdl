import importlib

from abc import ABCMeta, abstractmethod

class NetworkBase(metaclass=ABCMeta):
    def __init__(self, inputs, outputs, constraints, weights_conf, backend):
        self.inputs = inputs
        self.outputs = outputs
        self.constraints = constraints
        self.weights_conf = weights_conf
        self.__backend = backend
        self.__import_net()

    def __import_net(self):
        mdl_root = 'fws'
        name = self.get_name()
        mdl_path = '{}.{}.{}'.format(mdl_root, self.__backend, name)
        mdl = importlib.import_module(mdl_path)
        self.network_func = getattr(mdl, name)

    @abstractmethod
    def get_name(self):
        raise NotImplementedError('must return the module name of the network')

    def load(self)
        return self.network_func(self.inputs, self.outputs, self.constraints, self.weights_conf)
