from typing import List

class Outputs:
    def __init__(self, out_classes_list: List[int]):
        '''initializer
        :param out_classes_list: list of the no. of classes. one element if one output layer.
        '''
        self.num_classes_list = out_classes_list
