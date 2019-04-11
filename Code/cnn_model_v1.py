# Import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cnn_datasets

# Save model
model_save_name= "cnn_v1_model/"

# Model parameters
image_width = 128
image_height = 128
image_channel = 3
image_types = 5

# Placeholder variables
# Placeholder variable for the input images
x_train = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='x_train')
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, image_types], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Function creating new convolution layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        # Add the biases to the results of the convolution.
        layer += biases
        return layer, weights

# Function creating pooling layer
def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return layer

# Function creating relu layer
def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        return layer

# Function creating fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        return layer

# Create Convolutional Neural Network
# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_train, num_input_channels= image_channel, filter_size=5, num_filters=6, name ="conv1")
print(layer_conv1)
# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")
print(layer_pool1)
# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")
print(layer_relu1)

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")
print(layer_conv2)
# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")
print(layer_pool2)
# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")
print(layer_relu2)

# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])
print(layer_flat)

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
print(layer_fc1)
# RelU layer 3
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")
print(layer_relu3)

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=image_types, name="fc2")
print(layer_fc2)

# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost function
# Use Cross entropy cost function
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

# Optimizer
# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the FileWriter
writer_train = tf.summary.FileWriter("cnn_v1_training_file_writer/")
writer_valid = tf.summary.FileWriter("cnn_v1_validation_file_writer/")

# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()


num_epochs = 50
batch_size = 100
dataset  = cnn_datasets.data_ultils()

# TensorFlow Session
with tf.Session() as sess:
    # Saver
    saver = tf.train.Saver(max_to_keep=4)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Add the model graph to TensorBoard
    writer_train.add_graph(sess.graph)
    # Loop over number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        train_accuracy = 0
        for batch in range(0, int(len(dataset.y_train)/batch_size)):
            # Get a batch of images and labels
            x_batch, y_true_batch = dataset.get_data_batch(batch_size)
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x_train: x_batch, y_true: y_true_batch}
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            # Generate summary with the current batch of data and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer_train.add_summary(summ, epoch*int(len(dataset.y_train)/batch_size) + batch)

        saver.save(sess, model_save_name)

        train_accuracy /= int(len(dataset.y_train)/batch_size)
        # Generate summary and validate the model on the entire validation set
        summ, vali_accuracy = sess.run([merged_summary, accuracy], feed_dict={x_train:dataset.x_test, y_true:dataset.y_test})
        writer_valid.add_summary(summ, epoch)
        end_time = time.time()
        
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
