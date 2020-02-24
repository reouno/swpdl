from importlib import import_module
from typing import Text

import utils.constants as const
from utils.string import to_snake_case
from model_utils.weights_conf import WeightsConf

class Network:
    '''network definition
    '''

    def __init__(
            self,
            snake_case_name: Text,
            inputs,
            outputs,
            constraints,
            weights_conf,
            backend):
        '''initializer
        :param snake_case_name: name of the network
        :param inputs: inputs definition
        :param outputs: outputs definition
        :param constraints: weight constraints and dropout settings
        :param weights_conf: weights configuration
        :param backend: backend framework
        '''
        self.network_name = to_snake_case(snake_case_name)
        self.inputs = inputs
        self.outputs = outputs
        self.constraints = constraints
        self.weights_conf = weights_conf
        self.backend = backend

    def load(self):
        '''load network and weights if necessary
        '''
        mdl = self.__import_module(self.network_name)
        # load the class
        net_func = getattr(mdl, self.network_name)
        model = net_func(
                self.inputs,
                self.outputs,
                self.constraints,
                self.weights_conf,
                )
        if self.weights_conf.init_type == const.WeightsType.LOAD:
            model = self.__load_weights(model)
        return model

    def __import_module(self, name):
        '''import network definition module dynamically
        '''
        mdl_root = 'fws'
        backend = const.backend_mdls[self.backend]
        mdl_path = '{}.{}.{}.{}'.format(mdl_root, backend, 'nets', self.network_name)
        return import_module(mdl_path)

    def __load_weights(self, model):
        '''load weights file by using the backend specific method
        '''
        module_name = const.backend_util_module
        mdl = self.__import_module(module_name)
        load_weights = getattr(mdl, const.load_weights_func_name)
        return load_weights(model, self.weights_conf.file_path)
