from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import keras
from keras import backend as K, optimizers

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import DepthwiseConv2D
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import ReLU
from keras.layers import GaussianDropout
from keras.layers import add
from keras.layers import concatenate
from keras.layers import multiply
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.resnet50 import preprocess_input
from keras_applications.imagenet_utils import decode_predictions
from keras.layers import Flatten
import tensorflow as tf
    
def _lambda_unstack(x):
  return tf.unstack(x,axis=1)
   
def _lambda_expand_dims(x):
  return tf.expand_dims(x,axis=1)

def MobileNetV2(inputs,
                alpha=0.25,
                depth_multiplier=1,
                weightdecay=0.0002,
                jitter=0.001):
                
    """MobileNetv1
    This function defines a MobileNetv1 architectures.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        alpha: Width Multiplier
        depth_multiplier: Resolution Multiplier
        classes: number of labels
        weightdecay: weight decay for last layer
    # Returns
        five MobileNetv2 model stages."""
    
    img_input=Input(shape=(inputs.shape[1],inputs.shape[2],inputs.shape[3]), name='DC_IC')
    inputs = img_input
#    DC_inputs = Input(shape=(DC.shape[1],DC.shape[2],DC.shape[3]), name='DC_input')
#    IC_inputs = Input(shape=(IC.shape[1],IC.shape[2],IC.shape[3]), name='IC_input')

#    inputs_DC = DC_inputs
#    inputs_IC = IC_inputs

    if jitter != 0:
      input_jitter = GaussianDropout(jitter)(img_input)
      print("input shape: ", input_jitter.get_shape())
      DC_inputs = Lambda(lambda x:x[:,0:8,:,:])(input_jitter)
      IC_inputs = Lambda(lambda x:x[:,8:27,:,:])(input_jitter)

      #DC_inputs, IC_inputs = tf.split(input_jitter, num_or_size_splits = [8,19], axis=1)
      print("done jitter", DC_inputs.get_shape(), IC_inputs.get_shape())
    DC_branch = subnet(DC_inputs, "DC", alpha) #, input_shape=(DC.shape[1],DC.shape[2],DC.shape[3]))
    IC_branch = subnet(IC_inputs, "IC", alpha) #, input_shape=(IC.shape[1],IC.shape[2],IC.shape[3]))
 
    #subnets have different inputs, bottleneck layers take dc/ic inputs seperately until this point
    '''
    IC_branch = Lambda(_lambda_unstack)(IC_branch)
    branches=[DC_branch]
    for branch in DC_branch:
      print("DC")
    for branch in IC_branch:
      print("1")
      branch = Lambda(_lambda_expand_dims)(branch)
      branches.append(branch)
    merge = concatenate(branches)
    '''
    merge = concatenate([DC_branch,IC_branch],axis=1)

    merge = _inverted_residual_block(merge, 64,  (3, 3), t=6, strides=2, n=4, alpha=alpha, block_id=7, name='merge') 
    merge = _inverted_residual_block(merge, 96,  (3, 3), t=6, strides=1, n=3, alpha=alpha, block_id=11, name='merge')
    merge = _inverted_residual_block(merge, 160, (3, 3), t=6, strides=2, n=3, alpha=alpha, block_id=14, name='merge') 
    merge = _inverted_residual_block(merge, 320, (3, 3), t=6, strides=1, n=1, alpha=alpha, block_id=17, name='merge')
    merge = _conv_block(merge, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, name='merge')

    merge = GlobalAveragePooling2D()(merge)

    merge = Dropout(0.4)(merge)
    merge = Dense(1024,activation='relu')(merge)
    merge = Dropout(0.4)(merge)

    out = Dense(1,
                use_bias=False,
                kernel_regularizer=l2(weightdecay),
                activation='linear',
                name='output')(merge)
    print("before Model")
    model = Model(inputs=inputs, outputs=out, name='mobilenetv2')
#    print(model)
    # load weights
    return model

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1, name=''):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, chans)`
            (with `channels_last` data format) or
            (chans, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name=name+'conv{}'.format(block_id))(inputs)
    x = BatchNormalization(axis=channel_axis, name=name+'conv{}_bn'.format(block_id))(x)
    return ReLU(6, name=name+'conv{}_relu'.format(block_id))(x)

def _bottleneck(inputs, filters, kernel, t, s, r=False, alpha=1.0, block_id=1, train_bn = False, name=''):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    filters = int(alpha * filters)
    x = _conv_block(inputs, tchannel, alpha, (1, 1), (1, 1),block_id=block_id, name=name)
    x = DepthwiseConv2D(kernel,
                    strides=(s, s),
                    depth_multiplier=1,
                    padding='same',
                    name=name+'conv_dw_{}'.format(block_id))(x)
    x = BatchNormalization(axis=channel_axis,name=name+'conv_dw_{}_bn'.format(block_id))(x)
    x = ReLU(6, name=name+'conv_dw_{}_relu'.format(block_id))(x)
    x = Conv2D(filters,
                    (1, 1),
                    strides=(1, 1),
                    padding='same',
                    name=name+'conv_pw_{}'.format(block_id))(x)
    x = BatchNormalization(axis=channel_axis, name=name+'conv_pw_{}_bn'.format(block_id))(x, training=train_bn)
    if r:
        x = add([x, inputs], name=name+'res{}'.format(block_id))
    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha, block_id, name=''):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """
    x = _bottleneck(inputs, filters, kernel, t, strides, False, alpha, block_id, name=name)
    for i in range(1, n):
        block_id += 1
        x = _bottleneck(x, filters, kernel, t, 1, True, alpha, block_id, name=name)
    return x

def subnet(x, name, alpha):
    x = _conv_block(x, 32, alpha, (3, 3), strides=(2, 2), block_id=0, name=name)
    x = _inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1, alpha=alpha, block_id=1, name=name)
    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2, alpha=alpha, block_id=2, name=name)
    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3, alpha=alpha, block_id=4, name=name)
    return x
