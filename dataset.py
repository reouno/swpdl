from importlib import import_module

import utils.constants as const

class Dataset:
    '''dataset used for training/test/validation
    '''

    def __init__(self, conf, logger):
        logger.info('Dataset initializer called')
        self.backend = conf.backend
        self.dataset = conf.dataset
        self.batch_size = conf.batch_size
        self.input_size = (conf.in_height_list[0], conf.in_width_list[0])
        self.test_data = conf.test_data
        self.validation_data = conf.validation_data
        self.gpu_conf = conf.gpu_conf

    def load(self):
        '''load dataset
        '''
        mdl = self.__import_dataset_module()
        dataset_class = getattr(mdl, 'Dataset')
        return dataset_class(
                self.dataset,
                self.batch_size,
                self.input_size,
                self.gpu_conf,
                test_data_dir=self.test_data,
                validation_data_dir=self.validation_data
                )

    def __import_dataset_module(self):
        mdl_path = 'fws.{}.dataset'.format(const.backend_mdls[self.backend])
        return import_module(mdl_path)
