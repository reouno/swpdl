import tensorflow as tf

from typing import Dict, Text

def load(name: Text, params: Dict):
    '''load optimizer
    :param name: name of the optimizer
    :param params: dict of all optimizer parameters
    '''
    if name.upper() == 'ADAM':
        return tf.optimizers.Adam(
                params['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False
                )
    elif name.upper() == 'RMSPROP':
        print('\noptimizer is RMSprop\n')
        return tf.optimizers.RMSprop(
                params['learning_rate'],
                rho=0.9,
                momentum=0.9,
                epsilon=1e-10,
                centered=True
                )
    else:
        raise RuntimeError('undefined optimizer name "{}"'.format(name))

