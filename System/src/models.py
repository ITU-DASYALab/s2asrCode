

import tensorflow as tf
# pylint: disable=no-name-in-module
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow import flags

import util

FLAGS = flags.FLAGS
 


flags.DEFINE_integer("conv_width",
    #8,  # Our
    11, # Deep speech 2 Default lowest
    "The width of the convolution")
flags.DEFINE_integer("conv_output", 
    #512,  # Our
    1280, # Deep speech 2 Default Lowest
    # Checkout Deep speech 2 page 9
    "The Convolution output size")
flags.DEFINE_boolean("activation", True, "If Activation is applied between layers")
flags.DEFINE_boolean("add_batch_norm", True, "If batch normal is applied between layers")
flags.DEFINE_integer("LSTM_size", 
    #1024, #Our 
    1510, # Deep speech 5 layers, Page 6
    "The size of the LSTM layers")
flags.DEFINE_integer("LSTM_Layer_count", 
    5, 
    "The number of LSTM layers")
flags.DEFINE_integer("stride", 2, "The amount of stride for the convolution layer")
flags.DEFINE_boolean("cudnn_lstm", False, "Use CUDNN optimized LSTM")

flags.DEFINE_string("activation_function", "relu", "The activation function in the standard layers.")
flags.DEFINE_boolean("custom_cudnn", False, "Use the custom CUDNN LSTM with standard components between layers")
flags.DEFINE_boolean("lstm_dropout", True, "a boolean specifying if there should be dropout on the lstm layers.")



class BaseModel(object):
    """Inherit from this class when implementing new models."""
    def create_model(self, unused_model_input, **unused_params):
        raise NotImplementedError()


class LogisticModel(BaseModel):

    def create_model(self, model_input, output_size, training=False, **unused_params):

        conv_1  = conv_layer(
            model_input,
            [FLAGS.conv_width, model_input.get_shape()[-1], FLAGS.conv_output], 
            "conv_1",
            training,
            FLAGS.stride)

        ff_1 = feed_forward_layer(conv_1, output_size, "ff1", training)

        output = feed_forward_layer(ff_1, output_size, "output", training, standard=False)

        return {"predictions": output}



class DeepSpeech1(BaseModel):

    def create_model(self, model_input, output_size, training=False, **unused_params):

        conv_1  = conv_layer(
            model_input,
            [FLAGS.conv_width, model_input.get_shape()[-1], FLAGS.conv_output], 
            "conv_1",
            training,
            FLAGS.stride)

        ff_1 = feed_forward_layer(conv_1, output_size, "ff1", training)
        ff_2 = feed_forward_layer(ff_1, output_size, "ff2", training)

        lstm_outputs = lstm_layers(ff_2, training)
        
        output = feed_forward_layer(lstm_outputs, output_size, "output", training, standard=False)

        return {"predictions": output}

# Original width: 6
# Original stride: 2
# Original number of LSTM layers: 5
class StreamSpeechM33(BaseModel):

    def create_model(self, model_input, output_size, training=False, **unused_params):
        conv_1  = conv_layer(
            model_input,
            [FLAGS.conv_width, model_input.get_shape()[-1], FLAGS.conv_output],
            "conv_1",
            training,
            FLAGS.stride)

        lstm_outputs = lstm_layers(conv_1, training)

        output = feed_forward_layer(lstm_outputs, output_size, "output", training, standard=False)
        
        return {"predictions": output}

# Standard Feed Forward type layer.
class M34(BaseModel):
    def create_model(self, model_input, output_size, training=False, **unused_params):

        print(model_input)
        conv_1  = conv_layer(
            model_input,
            [FLAGS.conv_width, model_input.get_shape()[-1], FLAGS.conv_output],
            "conv_1",
            training,
            FLAGS.stride)

        with tf.compat.v1.variable_scope("ff"):
            w1 = tf.compat.v1.get_variable( 
                "w1" , [conv_1.get_shape()[-1], output_size], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.compat.v1.get_variable( 
                "b1" , [1, output_size], initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(conv_1, w1) + b1
        print(output)
        return {"predictions": output} 

# Helper methods

def feed_forward_layer(input, size, name, training, standard = True, initializer = initializers.xavier_initializer()):
    with tf.compat.v1.variable_scope(name):
        ff = slim.fully_connected(
            input,
            size,
            weights_regularizer=slim.l2_regularizer(1e-8),
            biases_regularizer= slim.l2_regularizer(1e-8),
            weights_initializer=initializer)

        if standard:
            ff = standard_components(ff, training)
        return ff

def conv_layer(
        input, 
        size,  
        name, 
        training,
        stride,
        padding= "VALID"):
    '''
    padding can be VALID or SAME
    '''
    with tf.compat.v1.variable_scope(name):
        convolution = tf.compat.v1.get_variable( 
            name , size, initializer=tf.contrib.layers.xavier_initializer())

        output = tf.nn.conv1d(
            input, 
            convolution,
            use_cudnn_on_gpu= True,
            stride = stride, 
            padding=padding)

        output = standard_components(output, training)

    return output 

def lstm_layers(input, training):
    if FLAGS.cudnn_lstm:
        return lstm_layers_cudnn(input, training)
    
    LSTMs = []
    layer = lstm_layer(input, FLAGS.LSTM_size, "lstm_1", training)
    LSTMs.append(layer)
    
    for x in range(2, FLAGS.LSTM_Layer_count + 1 ):
        layer = lstm_layer(
            LSTMs[-1][0], FLAGS.LSTM_size, "lstm_" + str(x), training)
        LSTMs.append(layer)
    return LSTMs[-1][0]

def lstm_layer(input, size, name, training):
    with tf.compat.v1.variable_scope(name):

        fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
            size, 
            forget_bias = 1.,
            use_peepholes = True,
            name=name + "_cell")

        output, state = tf.nn.dynamic_rnn(fw_cell, input, 
            dtype=tf.float32, swap_memory=True)

        output = standard_components(output, training, is_lstm_layer=True)
    return output, state

def lstm_layers_cudnn(input, training):
    if FLAGS.custom_cudnn:
        lstm_layers = []
        lstm_outputs = lstm_layer_cudnn(input, FLAGS.LSTM_size, "lstm_1", training, 1, 0.)
        standardized_layer1_output = standard_components(lstm_outputs, training, is_lstm_layer= True)
        lstm_layers.append(standardized_layer1_output)

        for x in range(2, FLAGS.LSTM_Layer_count + 1 ):
            layer = lstm_layer_cudnn(lstm_layers[-1], FLAGS.LSTM_size, "lstm_" + str(x), training, 1, 0.)
            with tf.compat.v1.variable_scope("lstm_" + str(x)):
                standardized_layerx_output = standard_components(layer, training)
            lstm_layers.append(standardized_layerx_output)
        return lstm_layers[-1]	
    else:
        lstm_outputs = lstm_layer_cudnn(input, FLAGS.LSTM_size, "cudnn_lstm", training, FLAGS.LSTM_Layer_count, FLAGS.dropout)
        return lstm_outputs

def lstm_layer_cudnn(input, size, name, training, num_layers, dropout):
    with tf.compat.v1.variable_scope(name):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=size,
            direction='unidirectional',
            dtype=tf.float32,
            bias_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            dropout=dropout if training else 0.,
            name=name + "_cell"
        )
        lstm.build(input.get_shape())
        lstm_outputs, lstm_output_states = lstm(input, training=training)
        return lstm_outputs

def standard_components(tensor, training, is_lstm_layer = False):
    if FLAGS.activation:
        activation_function = util.find_class_by_name(FLAGS.activation_function, [tf.compat.v1.nn])
        tensor = activation_function(tensor)
    if FLAGS.add_batch_norm:
        #tensor = tf.layers.batch_normalization(tensor, training=training, name= "norm")
        tensor = slim.batch_norm(
            tensor,
            center=True,
            scale=True,
            is_training=training)

    make_dropout = not is_lstm_layer or FLAGS.lstm_dropout
    if FLAGS.dropout > 0.0001 and make_dropout:
        dropout_rate = FLAGS.dropout if training else 0.
        dropout_probability = tf.compat.v1.placeholder_with_default(dropout_rate, shape=(), name= "dropout")
        tensor = tf.nn.dropout(tensor, rate=dropout_probability)
    return tensor