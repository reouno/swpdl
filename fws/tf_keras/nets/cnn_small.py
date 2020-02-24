from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, ReLU
from tensorflow.keras.models import Model
from typing import List

def cnn_small(inputs, outputs, constraints, weights_conf):
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

    # conv1
    x = Conv2D(32, 3)(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D()(x)

    # conv2
    x = Conv2D(64, 3)(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D()(x)

    # fcn
    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU(max_value=6)(x)

    # output
    x = Dense(out_units, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)
    return model
