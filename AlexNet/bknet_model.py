# Import library
import tensorflow as tf
import time
from dataset_loader import DatasetLoader
import network_config as config
from cnn_layers import Layers
import os

# Datasets
dataset  = DatasetLoader()

# Layers
layers = Layers()

# Save model
model_path_name= "bknet_model/"

# Config trainning
NUM_EPOCHS = 50
BATCH_SZE = 16

# Model parameters
image_width = config.image_width
image_height = config.image_height
image_channel = config.image_channel
image_types = dataset.get_num_types()

# Placeholder variable for the input images
x_train = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='x_train')
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, image_types], name='y_true')

# conv11 | relu11 | conv12 | relu12 | maxpool1
layer_conv1, weights_conv11= layers.new_conv_layer(input_tensor=x_train, input_channel= image_channel, 
filter_size=3, filter_num=32, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv11")
layer_conv1 = layers.new_relu_layer(layer_conv1, name="relu11")
print(layer_conv1)

layer_conv1, weights_conv12 = layers.new_conv_layer(input_tensor=layer_conv1, input_channel= 32, 
filter_size=3, filter_num=32, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv12")
layer_conv1 = layers.new_relu_layer(layer_conv1, name="relu12")
print(layer_conv1)

layer_conv1 = layers.new_pool_layer(input_tensor=layer_conv1, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool1")
print(layer_conv1)

# conv21 | relu21 | conv22 | relu22 | maxpool2
layer_conv2, weights_conv22 = layers.new_conv_layer(input_tensor=layer_conv1, input_channel= 32, 
filter_size=3, filter_num=64, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv21")
layer_conv2 = layers.new_relu_layer(layer_conv2, name="relu21")
print(layer_conv2)

layer_conv2, weights_conv21 = layers.new_conv_layer(input_tensor=layer_conv2, input_channel= 64, 
filter_size=3, filter_num=64, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv22")
layer_conv2 = layers.new_relu_layer(layer_conv2, name="relu22")
print(layer_conv2)

layer_conv2 = layers.new_pool_layer(input_tensor=layer_conv2, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool2")
print(layer_conv2)

# conv31 | relu31 | conv32 | relu32 | maxpool3
layer_conv3, weights_conv31 = layers.new_conv_layer(input_tensor=layer_conv2, input_channel= 64, 
filter_size=3, filter_num=128, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv31")
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu31")
print(layer_conv3)

layer_conv3, weights_conv32 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 128, 
filter_size=3, filter_num=128, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv32")
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu32")
print(layer_conv3)

layer_conv3 = layers.new_pool_layer(input_tensor=layer_conv3, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool3")
print(layer_conv3)

# conv41 | relu41 | conv42 | relu42 | conv43 | relu43 | maxpool4
layer_conv4, weights_conv41 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 128, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv41")
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu41")
print(layer_conv4)

layer_conv4, weights_conv42 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 256, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv42")
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu42")
print(layer_conv4)

layer_conv4 = layers.new_pool_layer(input_tensor=layer_conv4, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool4")
print(layer_conv4)

# conv5 | relu5 | maxpool5
layer_conv5, weights_conv5 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 256, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv5")
layer_conv5 = layers.new_relu_layer(layer_conv5, name="relu5")
print(layer_conv5)

layer_conv5 = layers.new_pool_layer(input_tensor=layer_conv5, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="SAME",name="pool5")
print(layer_conv5)

# Flatten Layer
num_features = layer_conv5.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_conv5, [-1, num_features])
print(layer_flat)

# Fully-Connected Layer 1
layer_fc1 = layers.new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
print(layer_fc1)
# RelU layer 4
layer_relu4 = layers.new_relu_layer(layer_fc1, name="relu4")
print(layer_relu4)

# Fully-Connected Layer 2
layer_output = layers.new_fc_layer(input=layer_relu4, num_inputs=128, num_outputs=image_types, name="fc2")
print(layer_output)

# Use Softmax function to normalize the output
with tf.variable_scope("softmax"):
    y_pred = tf.nn.softmax(layer_output)

# Use Cross entropy cost function
with tf.name_scope("cross_cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_output, labels=y_true)
    cost = tf.reduce_mean(cross_entropy, name="cross_cost")
    tf.summary.scalar("cross_cost", cost)

# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_true, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(model_path_name + "train")
writer_valid = tf.summary.FileWriter(model_path_name + "valid")
train_num_loop = 0
valid_num_loop = 0
print('Run `tensorboard --logdir=%s` to see the results.' % model_path_name)

# TensorFlow Session
with tf.Session() as sess:
    # Saver
    saver = tf.train.Saver(max_to_keep=4)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # # create a saver object to load the model
    # saver = tf.train.import_meta_graph(os.path.join(model_path_name, 'model.ckpt.meta'))
    # # restore the model from our checkpoints folder
    # saver.restore(sess, os.path.join(model_path_name, 'model.ckpt'))
    # Add the model graph to TensorBoard
    writer_train.add_graph(sess.graph)
    # Loop over number of epochs
    for epoch in range(NUM_EPOCHS):
        print("training epoch ", epoch)
        start_time = time.time()
        train_accuracy = 0
        for batch in range(0, int(dataset.get_num_train()/BATCH_SZE)):
            # Get a batch of images and labels
            dataset.load_datasets_random(config.training_buffer)
            x_batch, y_batch = dataset.load_batch_dataset(BATCH_SZE)
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

        train_accuracy /= int(dataset.get_num_train()/BATCH_SZE)
        # Generate summary and validate the model on the entire validation set
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
