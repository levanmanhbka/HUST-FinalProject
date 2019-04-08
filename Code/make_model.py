from dataset_handler import load_datasets
import tensorflow as tf

x_train, y_train, x_test, y_test = load_datasets()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#------------------------------- DEFINE PARAMETERS --------------------
height = 128
width = 128
color_channels = 3
number_of_classes = 4
#create Placeholders for images and labels
images_ph=tf.placeholder(tf.float32,shape=[None,height,width,color_channels])
labels_ph=tf.placeholder(tf.float32,shape=[None,number_of_classes])

#-------------------------------- CREATE ELEMENTS ---------------------
class model_unit:
    def add_weights(self,shape):
        # a common method to create all sorts of weight connections
        # takes in shapes of previous and new layer as a list e.g. [2,10]
        # starts with random values of that shape.
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

    def add_biases(self,shape):
        # a common method to add create biases with default=0.05
        # takes in shape of the current layer e.g. x=10
        return tf.Variable(tf.constant(0.05, shape=shape))

    def conv_layer(self,layer, kernel, input_depth, output_depth, stride_size):
        #convolution occurs here.
        #create weights and biases for the given layer shape
        weights = self.add_weights([kernel, kernel, input_depth, output_depth])
        biases = self.add_biases([output_depth])
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1] mostly
        stride = [1, stride_size, stride_size, 1]
        #does a convolution scan on the given image
        layer = tf.nn.conv2d(layer, weights, strides=stride, padding='SAME') + biases
        return layer

    def pooling_layer(self,layer, kernel_size, stride_size):
        # basically it reduces the complexity involved by only taking the important features alone
        # many types of pooling is there.. average pooling, max pooling..
        # max pooling takes the maximum of the given kernel
        #kernel=[image_jump,rows,columns,depth]
        kernel = [1, kernel_size, kernel_size, 1]
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,2,2,1] mostly
        stride = [1, stride_size, stride_size, 1]
        return tf.nn.max_pool(layer, ksize=kernel, strides=stride, padding='SAME')
    
    def flattening_layer(self,layer):
        #make it single dimensional
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]), new_size

    def fully_connected_layer(self,layer, input_shape, output_shape):
        #create weights and biases for the given layer shape
        weights = self.add_weights([input_shape, output_shape])
        biases = self.add_biases([output_shape])
        #most important operation
        layer = tf.matmul(layer,weights) + biases  # mX+b
        return layer

    def activation_layer(self,layer):
        # we use Rectified linear unit Relu. it's the standard activation layer used.
        # there are also other layer like sigmoid,tanh..etc. but relu is more efficent.
        # function: 0 if x<0 else x.
        return tf.nn.relu(layer)


#-------------------------------- CREATE NETWORK---------------------
def generate_model():
    model = model_unit()
    print("level 1 convolution")
    network=model.conv_layer(images_ph,5,3,16,1)
    network=model.pooling_layer(network,5,2)
    network=model.activation_layer(network)
    print(network)

    print("level 2 convolution")
    network=model.conv_layer(network,4,16,32,1)
    network=model.pooling_layer(network,4,2)
    network=model.activation_layer(network)
    print(network)

    print("level 3 convolution")
    network=model.conv_layer(network,3,32,64,1)
    network=model.pooling_layer(network,3,2)
    network=model.activation_layer(network)
    print(network)

    print("flattening layer")
    network, features=model.flattening_layer(network)
    print(network)

    print("fully connected layer 1")
    network=model.fully_connected_layer(network,features,1024)
    network=model.activation_layer(network)
    print(network)

    print("fully connected layer 2")
    network=model.fully_connected_layer(network,1024,number_of_classes)
    print(network)

    print("output layer")
    network = tf.nn.softmax(network)
    print(network)

    return network


network = generate_model()

#-------------------------------- TRAINING NETWORK---------------------
#find error like squared error but better
cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=labels_ph)

#now minize the above error
#calculate the total mean of all the errors from all the nodes
cost=tf.reduce_mean(cross_entropy)

#Now backpropagate to minimise the cost in the network.
optimizer=tf.train.AdamOptimizer().minimize(cost)

session=tf.Session()

