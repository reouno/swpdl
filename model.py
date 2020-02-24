from model_utils.constraints import Constraints
from model_utils.inputs import Inputs
from model_utils.outputs import Outputs
from model_utils.weights_conf import WeightsConf
from network import Network

class Model:
    '''load deep learning model
    '''

    def __init__(self, conf, logger):
        logger.info('Model initializer called')
        self.name = conf.network
        self.inputs = Inputs(
                conf.in_height_list,
                conf.in_width_list,
                conf.in_channels_list
                )
        self.outputs = Outputs(conf.out_classes_list)
        self.constraints = Constraints(conf)
        self.weights_conf = WeightsConf(conf.weights_type, conf.weights_path)
        self.backend = conf.backend

    def load(self):
        return Network(
                self.name,
                self.inputs,
                self.outputs,
                self.constraints,
                self.weights_conf,
                self.backend
                ).load()
