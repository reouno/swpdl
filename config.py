import os
import shutil
import yaml

from argparse import ArgumentParser
from importlib import import_module
from typing import List, Text

import utils.constants as c
import utils.user_interfaces as ui

class Config:
    '''configurations
    '''

    def __init__(self):
        self.__swpdl_root = os.path.split(__file__)[0]
        self.a = self.parse_args() # from command line arguments
        self.conf = self.load_conf_file(self.from_swpdl_root(self.a['conf']))
        self.load()

    def parse_args(self):
        psr = arg_parser()
        a_dict = vars(psr.parse_args())
        return a_dict

    def load_conf_file(self, file_path):
        return read_yaml(file_path)

    def makedirs_after_confirmation(self, dir_path: Text):
        if os.path.exists(dir_path):
            confirmation_msg = 'Delete {}'.format(dir_path)
            refused_msg = 'Cannot delete the directory. Delete it or specify another file path and try again.'
            yes = ui.get_user_confirmation(confirmation_msg, refused_msg)
            if yes:
                shutil.rmtree(dir_path)
            else:
                exit()

        os.makedirs(dir_path)

    def load(self):
        # basic conf
        self.backend = set_backend(self.get_value('backend').upper())
        self.gpu_conf = self.set_gpu_conf(self.get_value('gpus'))
        self.save_dir = self.get_value('save_dir')
        self.makedirs_after_confirmation(self.save_dir)

        # dataset
        self.dataset = self.get_value('dataset')
        self.test_data = self.get_value('test_data')
        self.validation_data = self.get_value('validation_data')
        self.num_train_samples = None # need update after loading dataset
        self.num_test_samples = None # need update after loading dataset
        self.num_validation_samples = None # need update after loading dataset

        # network
        self.network = self.get_value('network')
        self.in_height_list = self.get_value('in_heights')
        self.in_width_list = self.get_value('in_widths')
        self.in_channels_list = self.get_value('in_channels')
        assert len(self.in_height_list) == len(self.in_width_list) == len(self.in_channels_list)
        self.out_classes_list = self.get_value('classes')
        self.set_weights_confs(self.get_value('weights'))
        self.dropout_keep_prob = self.get_value('dropout_keep_prob')
        self.l2_lambda = self.get_value('l2_lambda')

        # optimization
        self.optimizer = self.get_value('optimizer')
        self.learning_rate = self.get_value('learning_rate')
        self.learning_rate_decay = self.get_value('learning_rate_decay')

        # training
        self.batch_size = self.get_value('batch_size')
        self.num_epochs = self.get_value('num_epochs')
        self.train_steps_per_epoch = None # need update after loading dataset
        self.test_steps_per_epoch = None # need update after loading dataset
        self.validation_steps_per_epoch = None # need update after loading dataset

    def from_swpdl_root(self, path: Text):
        return os.path.join(self.__swpdl_root, path)

    def get_value(self, target):
        '''get config value
        1. try to get from command line arguments if defined
        2. get from config file
        '''
        if (target in self.a) and (self.a[target] is not None):
            return self.a[target]
        else:
            return self.conf[target]

    def set_gpu_conf(self, gpus: List[int]):
        print(self.backend)
        print(c.backend_mdls[self.backend])
        gpu_conf_mdl = import_module('fws.{}.gpu_conf'.format(c.backend_mdls[self.backend]))
        conf_class = getattr(gpu_conf_mdl, 'GPUConf')
        return conf_class(gpus)

    def set_weights_confs(self, weights: Text):
        if weights is None or weights == '':
            self.weights_type = c.WeightsType.RANDOM
            self.weights_path = None
        elif weights == 'imagenet':
            self.weights_type = c.WeightsType.PREPARED
            self.weights_path = None
        elif os.path.isfile(weights):
            self.weights_type = c.WeightsType.LOAD
            self.weights_path = weights
        else:
            raise RuntimeError('invalid weights type, "{}"'.format(wt))

    def set_num_samples_and_steps_per_epoch(
            self,
            num_train: int,
            num_test: int,
            num_validation: int
            ):
        self.num_train_samples = num_train
        self.num_test_samples = num_test
        self.num_validation_samples = num_validation
        if self.num_train_samples is not None:
            self.train_steps_per_epoch = self.num_train_samples // self.batch_size
        if self.num_test_samples is not None:
            self.test_steps_per_epoch = self.num_test_samples // self.batch_size
        if self.num_validation_samples is not None:
            self.validation_steps_per_epoch = self.num_validation_samples // self.batch_size

def arg_parser():
    psr = ArgumentParser()
    psr.add_argument('--conf', help='config yaml file path', default='config.yaml')
    # basic
    psr.add_argument('--gpus', help='device number list of gpus', nargs='+', type=int, required=False)
    psr.add_argument('--save_dir', help='directory to save all results and logs', required=False)
    # dataset
    psr.add_argument('--dataset', help='dataset name or directory path.', required=False)
    psr.add_argument('--test_data', help='test dataset directory path.', required=False)
    psr.add_argument('--validation_data', help='validation dataset directory path.', required=False)
    # network
    psr.add_argument('--network', help='name of network', required=False)
    psr.add_argument('--dropout_keep_prob', help='keep probability of dropout', required=False)
    psr.add_argument('--l2_lambda', help='lambda of l2 regularization for weight decay', required=False)
    # optimization
    psr.add_argument('--optimizer', help='name of the optmizer', required=False)
    psr.add_argument('--learning_rate', help='learning_rate', required=False)
    psr.add_argument('--learning_rate_decay', help='learning_rate_decay', required=False)
    # training
    psr.add_argument('--batch_size', help='global batch size', type=int, required=False)
    psr.add_argument('--num_epochs', help='the number of epochs for training', type=int, required=False)

    return psr

def read_yaml(file_path: Text):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def set_backend(be):
    if be == 'CAFFE':
        return c.Backend.CAFFE
    elif be == 'KERAS':
        return c.Backend.KERAS
    elif be == 'PYTORCH':
        return c.Backend.PYTORCH
    elif be == 'TENSORFLOW':
        return c.Backend.TENSORFLOW
    elif be == 'TF_KERAS':
        return c.Backend.TF_KERAS
    else:
        raise RuntimeError('invalid backend, "{}"'.format(be))
