import tensorflow as tf

from typing import List

class GPUConf:
    def __init__(self, gpu_devices: List[int]):
        self.gpu_devices = gpu_devices
        self.dev_strs = list(map(lambda x:'/gpu:{}'.format(x), self.gpu_devices))
        # If the list of devices is not specified in the
        # `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
        self.strategy = tf.distribute.MirroredStrategy(self.dev_strs)
        #self.strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
        #self.strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"])
        print('self.strategy.num_replicas_in_sync',self.strategy.num_replicas_in_sync)
