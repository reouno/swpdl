from enum import Enum

#### for backends ####

class Backend(Enum):
    CAFFE = 1
    KERAS = 2
    PYTORCH = 3
    TENSORFLOW = 4
    TF_KERAS = 5

backend_mdls = {
        Backend.CAFFE: 'caffe',
        Backend.KERAS: 'keras',
        Backend.PYTORCH: 'pytorch',
        Backend.TENSORFLOW: 'tensorflow',
        Backend.TF_KERAS: 'tf_keras'
        }

backend_util_module = 'utils'
load_weights_func_name = 'load_weights'


#### for weights ####

class WeightsType(Enum):
    RANDOM = 1
    PREPARED = 2
    LOAD = 3
