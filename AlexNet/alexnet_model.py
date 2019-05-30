import tensorflow as tf
import time
from dataset_loader import DatasetLoader
from cnn_layers import Layers
import network_config as config
import os

# Layer
layers = Layers()

# Datasets
dataset  = DatasetLoader()

# General parameters of the model
NUM_EPOCHS = 50
BATCH_SIZE = 128
DROPOUT_KEEP_PROB = 0.5
K_BIAS = 2
N_DEPTH_RADIUS = 5
ALPHA = 1e-4
BETA = 0.75

# Global dataset dictionary
image_width = config.image_width
image_height = config.image_height
image_channel = config.image_channel
image_types = dataset.get_num_types()

model_path_name= "alex_model/"

# Placeholder variable for the input images
x_train = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='x_train')
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, image_types], name='y_true')

# Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
print("Convolutional Layer 1")
layer_conv1, weights_conv1 = layers.new_conv_layer(input_tensor=x_train, input_channel= image_channel, 
filter_size=11, filter_num=96, filter_stride=[1, 4, 4, 1], filter_padding="VALID",name ="conv1")
print(layer_conv1)
# Normalize layer 1
layer_conv1 = tf.nn.lrn(layer_conv1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
print(layer_conv1)
# Pooling layer 1
layer_conv1 = layers.new_pool_layer(input_tensor=layer_conv1, 
ker_size=[1, 3, 3, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool1")
print(layer_conv1)
# RelU layer 1
layer_conv1 = layers.new_relu_layer(layer_conv1, name="relu1")
print(layer_conv1)

# Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
print("Convolutional Layer 2")
layer_conv2, weights_conv2 = layers.new_conv_layer(input_tensor=layer_conv1, input_channel= 96, 
filter_size=5, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv2")
print(layer_conv2)
# Normalize layer 2
layer_conv2 = tf.nn.lrn(layer_conv2, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
print(layer_conv2)
# Pooling layer 2
layer_conv2 = layers.new_pool_layer(input_tensor=layer_conv2, 
ker_size=[1, 3, 3, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool2")
print(layer_conv2)
# RelU layer 2
layer_conv2 = layers.new_relu_layer(layer_conv2, name="relu2")
print(layer_conv2)

# Convolution Layer 3 | ReLU
print("Convolutional Layer 3")
layer_conv3, weights_conv3 = layers.new_conv_layer(input_tensor=layer_conv2, input_channel= 256, 
filter_size=3, filter_num=384, filter_stride=[1, 1, 1, 1], filter_padding="SAME", name ="conv3")
print(layer_conv3)
# RelU layer 3
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu3")
print(layer_conv3)

# Convolution Layer 4 | ReLU
print("Convolutional Layer 4")
layer_conv4, weights_conv4 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 384, 
filter_size=3, filter_num=384, filter_stride=[1, 1, 1, 1], filter_padding="SAME", name ="conv4")
print(layer_conv4)
# RelU layer 3
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu4")
print(layer_conv4)

# Convolution Layer 5 | ReLU | Max Pooling
print("Convolutional Layer 5")
layer_conv5, weights_conv5 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 384, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME", name ="conv5")
print(layer_conv5)
# RelU layer 3
layer_conv5 = layers.new_relu_layer(layer_conv5, name="relu5")
print(layer_conv5)
# Pooling layer 2
layer_conv5 = layers.new_pool_layer(input_tensor=layer_conv5, 
ker_size=[1, 3, 3, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool5")
print(layer_conv5)

# Flatten Layer
print("flattent layer")
num_features = layer_conv5.get_shape()[1:4].num_elements()
feature_map = tf.reshape(layer_conv5, [-1, num_features])
print(feature_map)

# Fully Connected Layer 1 | Dropout
fc_layer_1 = layers.new_fc_layer(input=feature_map, num_inputs=num_features, num_outputs= 4096, name="fc_layer1")
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=DROPOUT_KEEP_PROB)
print(fc_layer_1)

# Fully Connected Layer 2 | Dropout
fc_layer_2 = layers.new_fc_layer(input=fc_layer_1, num_inputs=4096, num_outputs= 4096, name="fc_layer2")
fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=DROPOUT_KEEP_PROB)
print(fc_layer_2)

# Fully Connected Layer 3 | Softmax
fc_layer_3 = layers.new_fc_layer(input=fc_layer_1, num_inputs=4096, num_outputs= image_types, name="fc_layer2")
print(fc_layer_3)


#--------------------------- Training model -------------------------#
# Use Softmax function to normalize the output
with tf.variable_scope("softmax"):
    y_pred = tf.nn.softmax(fc_layer_3)

# Use Cross entropy cost function
with tf.name_scope("cross_cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc_layer_3, labels= y_true), name="cross_cost")
    tf.summary.scalar("cross_cost", cost)

# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(model_path_name + "train")
writer_valid = tf.summary.FileWriter(model_path_name + "valid")
train_num_loop = 0
valid_num_loop = 0
print('Run `tensorboard --logdir=%s` to see the results.' % model_path_name)

with tf.Session() as sess:
    # Saver
    saver = tf.train.Saver(max_to_keep=4)
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # # create a saver object to load the model
    # saver = tf.train.import_meta_graph(os.path.join(model_path_name, 'model.ckpt.meta'))
    # # restore the model from our checkpoints folder
    # saver.restore(sess, os.path.join(model_path_name, 'model.ckpt'))
    # Add the model graph to TensorBoard
    writer_train.add_graph(sess.graph)
    # Loop over number of eporchs
    for epoch in range(NUM_EPOCHS):
        print("training epoch ", epoch)
        dataset.load_datasets_random(3000)
        start_time = time.time()
        train_accuracy = 0.0
        for batch in range(0, int(dataset.get_num_train()/BATCH_SIZE)):
            # Get a batch of images and labels
            x_batch, y_batch = dataset.load_batch_dataset(BATCH_SIZE)
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x_train: x_batch, y_true: y_batch}
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            # Generate summary with the current batch of data and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer_train.add_summary(summ, train_num_loop)
            train_num_loop += 1
        
        saver.save(sess, os.path.join(model_path_name, "model.ckpt"))
        
        train_accuracy = train_accuracy / (dataset.get_num_train()/BATCH_SIZE)
        # Validate the model on the entire validation set
        print("validating epoch ", epoch)
        vali_accuracy = 0
        num_test_patch = 0
        dataset.reset_data_test_index()
        while dataset.load_data_test_continuos():
            summ, vali_accuracy_temp = sess.run([merged_summary, accuracy], feed_dict={x_train:dataset.x_test, y_true:dataset.y_test})
            writer_valid.add_summary(summ, valid_num_loop)
            vali_accuracy += vali_accuracy_temp
            num_test_patch += 1.0
            valid_num_loop += 1
        vali_accuracy /= num_test_patch

        end_time = time.time()
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\t- Training Accuracy:\t{}".format(train_accuracy))
        print("\t- Validation Accuracy:\t{}".format(vali_accuracy))
        print("")
