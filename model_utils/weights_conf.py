import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from typing import Text

import utils.constants as const

class WeightsConf:
    '''weights config
    '''

    def __init__(
            self,
            weights_type: const.WeightsType,
            weights_path: Text
            ):
        '''inirializer
        :param weights_type: method of weights initialization
        :param weights_path: weights file path
        '''
        self.init_type = weights_type
        self.file_path = weights_path
        self.__init_vars()

    def __init_vars(self):
        if self.init_type == const.WeightsType.PREPARED:
            self.keras_weights = 'imagenet' # for keras
            self.tf_keras_weights = 'imagenet' # for tf_keras
            self.pytorch_pretrained = True # for pytorch
        else:
            self.keras_weights = None # for keras
            self.tf_keras_weights = None # for tf_keras
            self.pytorch_pretrained = False # for pytorch
