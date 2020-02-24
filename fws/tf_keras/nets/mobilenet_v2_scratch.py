from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Add, Conv2D, Dense, DepthwiseConv2D, Dropout, GlobalAveragePooling2D, Flatten, Input, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from typing import List


def mobilenet_v2_scratch(inputs, outputs, constraints, weights_conf):
    '''mobilenet v2 defined from scrach
    implemented by referring the following implementation:
    refer: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

    :param inputs: input definition
    :param outputs: output definition
    :param constraints: constraints
    :param weights_conf: weights configuration
    '''

    # input shape in format (H, W, D)
    in_shape= [inputs.height_list[0], inputs.width_list[0], inputs.channels_list[0]]
    # no. of classes
    out_units= outputs.num_classes_list[0]
    # alpha value to adjust the no. of filters
    alpha= 1.0
    # lambda for L2 regularization
    weight_decay= constraints.l2_lambda
    if weight_decay is None:
        weight_decay = 0
    # keep probability of dropout
    dropout_keep_prob= constraints.dropout_keep_prob
    if dropout_keep_prob is None:
        dropout_keep_prob = 1.0
    #print('\n\nweight_decay =', weight_decay, '\n\n')

    # input
    input_img = Input(shape=in_shape)

    # first conv layer
    x = Conv2D(32, (3,3), strides=(2,2), padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            use_bias=False
            )(input_img)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(max_value=6)(x)

    # bottleneck block 1
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
            expansion=1, weight_decay=weight_decay, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
            expansion=6, weight_decay=weight_decay, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
            expansion=6, weight_decay=weight_decay, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
            expansion=6, weight_decay=weight_decay, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=10) # stride is 1 in this layer!!
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
            expansion=6, weight_decay=weight_decay, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
            expansion=6, weight_decay=weight_decay, block_id=13)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    # last conv layer
    x = Conv2D(last_block_filters, (1,1),
            kernel_regularizer=regularizers.l2(weight_decay),
            use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
            name='Conv_1_bn')(x)
    x = ReLU(max_value=6, name='out_relu')(x)

    # top layer
    x = GlobalAveragePooling2D()(x)
    if dropout_keep_prob < 1.0:
        x = Dropout(1.0 - dropout_keep_prob)(x)
    x = Dense(out_units, activation='softmax',
            kernel_regularizer=regularizers.l2(weight_decay),
            bias_regularizer=regularizers.l2(weight_decay),
            use_bias=True, name='Logits')(x)

    # create model
    model = Model(inputs=input_img,
            outputs=x,
            name='mobilenetv2_{0:2f}_{1}'.format(alpha, in_shape[0]))

    return model

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, weight_decay, block_id):
    in_channels = int(inputs.shape[-1])
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # expand (pointwise)
    if block_id:
        x = Conv2D(in_channels * expansion, (1,1), padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                use_bias=False)(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = ReLU(max_value=6)(x)
    else:
        pass

    # feature extraction (depthwise)
    x = DepthwiseConv2D((3,3), strides=stride, padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(max_value=6)(x)

    # projection (pointwise)
    # output channel size depends on 
    x = Conv2D(pointwise_filters, (1,1), padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add()([inputs, x])
    else:
        return x
