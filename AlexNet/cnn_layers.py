import tensorflow as tf

BN_EPSILON = 0.001

class Layers():
    # Function creating new convolution layer
    def new_conv_layer(self,input_tensor, input_channel, filter_size, filter_num, filter_stride, filter_padding, name):
        with tf.variable_scope(name):
            # Shape of the filter-weights for the convolution
            shape = [filter_size, filter_size, input_channel, filter_num]
            # Create new weights (filters) with the given shape
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
            # Create new biases, one for each filter
            biases = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="B")
            # TensorFlow operation for convolution
            layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=filter_stride, padding=filter_padding)
            # Add the biases to the results of the convolution.
            layer += biases
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            return layer, weights

    # Function creating pooling layer
    def new_pool_layer(self,input_tensor, ker_size, ker_stride, ker_padding,name):
        with tf.variable_scope(name):
            # TensorFlow operation for convolution
            layer = tf.nn.max_pool(value=input_tensor, ksize=ker_size, strides=ker_stride, padding=ker_padding)
            return layer

    # Function creating relu layer
    def new_relu_layer(self,input, name):
        with tf.variable_scope(name):
            # TensorFlow operation for convolution
            layer = tf.nn.relu(input)
            tf.summary.histogram("activations", layer)
            return layer

    # Function creating fully connected layer
    def new_fc_layer(self,input, num_inputs, num_outputs, name):
        with tf.variable_scope(name):
            # Create new weights and biases.
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.1), name="W")
            biases = tf.Variable(tf.constant(0.1, shape=[num_outputs]), name="B")
            # Multiply the input and weights, and then add the bias-values.
            layer = tf.matmul(input, weights) + biases
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", layer)
            return layer

    def new_batch_normalization_layer(self, input_layer, dimension, name = "bnl"):
        with tf.variable_scope(name):
            '''
            Helper function to do batch normalziation
            :param input_layer: 4D tensor
            :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
            :return: the 4D tensor after being normalized
            '''
            mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
            beta = tf.get_variable('beta', dimension, tf.float32,
                                    initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', dimension, tf.float32,
                                        initializer=tf.constant_initializer(1.0, tf.float32))
            bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

            return bn_layer


    def new_conv_bn_relu_layer(self, input_layer, filter_shape, stride, name = "conv_bn_relu"):
        with tf.variable_scope(name):
            '''
            A helper function to conv, batch normalize and relu the input tensor sequentially
            :param input_layer: 4D tensor
            :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
            :param stride: stride size for conv
            :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
            '''

            out_channel = filter_shape[-1]
            filter = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
            conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
            bn_layer = self.new_batch_normalization_layer(conv_layer, out_channel)

            output = tf.nn.relu(bn_layer)
            return output

    def new_residual_block(self, input_layer, output_channel, first_block=False, name = "id_block"):
        with tf.variable_scope(name):
            '''
            Defines a residual block in ResNet
            :param input_layer: 4D tensor
            :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
            :param first_block: if this is the first residual block of the whole network
            :return: 4D tensor.
            '''
            input_channel = input_layer.get_shape().as_list()[-1]

            # When it's time to "shrink" the image size, we use stride = 2
            if input_channel * 2 == output_channel:
                increase_dim = True
                stride = 2
            elif input_channel == output_channel:
                increase_dim = False
                stride = 1
            else:
                raise ValueError('Output and input channel does not match in residual blocks!!!')

            # The first conv layer of the first residual block does not need to be normalized and relu-ed.
            with tf.variable_scope('conv1_in_block'):
                if first_block:
                    filter = tf.Variable(tf.truncated_normal(shape=[3, 3, input_channel, output_channel], stddev=0.1), name="W")
                    conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 2, 2, 1], padding='SAME')
                else:
                    conv1 = self.new_conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channel], stride)

            with tf.variable_scope('conv2_in_block'):
                conv2 = self.new_conv_bn_relu_layer(conv1, [3, 3, output_channel, output_channel], 1)

            # When the channels of input layer and conv2 does not match, we add zero pads to increase the
            #  depth of input layers
            if increase_dim is True:
                pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
                padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
            else:
                padded_input = input_layer

            output = conv2 + padded_input
            return output