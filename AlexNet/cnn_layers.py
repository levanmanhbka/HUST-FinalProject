import tensorflow as tf

class Layers():
    # Function creating new convolution layer
    def new_conv_layer(self,input_tensor, input_channel, filter_size, filter_num, filter_stride, filter_padding, name):
        with tf.variable_scope(name) as scope:
            # Shape of the filter-weights for the convolution
            shape = [filter_size, filter_size, input_channel, filter_num]
            # Create new weights (filters) with the given shape
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
            # Create new biases, one for each filter
            biases = tf.Variable(tf.constant(0.05, shape=[filter_num]))
            # TensorFlow operation for convolution
            layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=filter_stride, padding=filter_padding)
            # Add the biases to the results of the convolution.
            layer += biases
            return layer, weights

    # Function creating pooling layer
    def new_pool_layer(self,input_tensor, ker_size, ker_stride, ker_padding,name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.max_pool(value=input_tensor, ksize=ker_size, strides=ker_stride, padding=ker_padding)
            return layer

    # Function creating relu layer
    def new_relu_layer(self,input, name):
        with tf.variable_scope(name) as scope:
            # TensorFlow operation for convolution
            layer = tf.nn.relu(input)
            return layer

    # Function creating fully connected layer
    def new_fc_layer(self,input, num_inputs, num_outputs, name):
        with tf.variable_scope(name) as scope:
            # Create new weights and biases.
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
            biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
            # Multiply the input and weights, and then add the bias-values.
            layer = tf.matmul(input, weights) + biases
            return layer