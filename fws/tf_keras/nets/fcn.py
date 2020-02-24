from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU
from tensorflow.keras.models import Model
from typing import List

def fcn(inputs, outputs, constraints, weights_conf):
    '''fcn
    :param inputs: input definition
    :param outputs: output definition
    :param constraints: constraints
    :param weights_conf: weights configuration
    '''

    in_shape = [inputs.height_list[0], inputs.width_list[0], inputs.channels_list[0]]
    hidden_layers = [128]
    out_units = outputs.num_classes_list[0]

    # input layer
    input_img = Input(shape=in_shape)
    x = input_img

    # hidden layers
    x = Flatten()(x)
    for num_units in hidden_layers:
        x = Dense(num_units)(x)
        x = ReLU(max_value=6)(x)

    # output layer
    x = Dense(out_units, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)
    return model
