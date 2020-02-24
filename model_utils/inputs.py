from typing import List

class Inputs:
    def __init__(
            self,
            in_height_list: List[int],
            in_width_list: List[int],
            in_channels_list: List[int]
            ):
        '''initializer
        :param in_height_list: list of input height. one element if one input layer.
        :param in_width_list: list of input width. one element if one input layer.
        :param in_channels_list: list of the no. of input channels. one element if one input layer.
        '''
        self.height_list = in_height_list
        self.width_list = in_width_list
        self.channels_list = in_channels_list
