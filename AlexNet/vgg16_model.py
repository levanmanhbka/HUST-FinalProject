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
model_path_name= "vgg16_model"
#model_path_name= config.root_path + "/code_project/vgg16_model" #for google colab

# Config trainning
NUM_EPOCHS = 50
BATCH_SIZE = 64

# Model parameters
image_width = config.image_width
image_height = config.image_height
image_channel = config.image_channel
image_types = dataset.get_num_types()

# Placeholder variable for the input images
x_train = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='x_train')
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, image_types], name='y_true')

# Conv_11
layer_conv1, weights_conv11 = layers.new_conv_layer(input_tensor=x_train, input_channel= image_channel, 
filter_size=3, filter_num=64, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv11")
# Relu_11
layer_conv1 = layers.new_relu_layer(layer_conv1, name="relu11")
# Conv_12
layer_conv1, weights_conv12 = layers.new_conv_layer(input_tensor=layer_conv1, input_channel= 64, 
filter_size=3, filter_num=64, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv12")
# Relu_12
layer_conv1 = layers.new_relu_layer(layer_conv1, name="relu12")
# Pooling_1
layer_conv1 = layers.new_pool_layer(input_tensor=layer_conv1, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool1")
print(layer_conv1)

# Conv_21
layer_conv2, weights_conv21 = layers.new_conv_layer(input_tensor=layer_conv1, input_channel= 64, 
filter_size=3, filter_num=128, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv21")
# Relu_21
layer_conv2 = layers.new_relu_layer(layer_conv2, name="relu21")
# Conv_22
layer_conv2, weights_conv22 = layers.new_conv_layer(input_tensor=layer_conv2, input_channel= 128, 
filter_size=3, filter_num=128, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv22")
# Relu_22
layer_conv2 = layers.new_relu_layer(layer_conv2, name="relu22")
# Pooling_2
layer_conv2 = layers.new_pool_layer(input_tensor=layer_conv2, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool2")
print(layer_conv2)

# Conv_31
layer_conv3, weights_conv31 = layers.new_conv_layer(input_tensor=layer_conv2, input_channel= 128, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv31")
# Relu_31
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu31")
# Conv_32
layer_conv3, weights_conv32 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 256, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv32")
# Relu_32
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu32")
# Conv_33
layer_conv3, weights_conv33 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 256, 
filter_size=3, filter_num=256, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv33")
# Relu_33
layer_conv3 = layers.new_relu_layer(layer_conv3, name="relu33")
# Pooling_3
layer_conv3 = layers.new_pool_layer(input_tensor=layer_conv3, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool3")
print(layer_conv3)


# Conv_41
layer_conv4, weights_conv41 = layers.new_conv_layer(input_tensor=layer_conv3, input_channel= 256, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv41")
# Relu_41
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu41")
# Conv_42
layer_conv4, weights_conv42 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 512, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv42")
# Relu_42
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu42")
# Conv_43
layer_conv4, weights_conv43 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 512, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv43")
# Relu_43
layer_conv4 = layers.new_relu_layer(layer_conv4, name="relu43")
# Pooling_4
layer_conv4 = layers.new_pool_layer(input_tensor=layer_conv4, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool4")
print(layer_conv4)

# Conv_51
layer_conv5, weights_conv51 = layers.new_conv_layer(input_tensor=layer_conv4, input_channel= 512, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv51")
# Relu_51
layer_conv5 = layers.new_relu_layer(layer_conv5, name="relu51")
# Conv_52
layer_conv5, weights_conv52 = layers.new_conv_layer(input_tensor=layer_conv5, input_channel= 512, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv52")
# Relu_52
layer_conv5 = layers.new_relu_layer(layer_conv5, name="relu52")
# Conv_53
layer_conv5, weights_conv53 = layers.new_conv_layer(input_tensor=layer_conv5, input_channel= 512, 
filter_size=3, filter_num=512, filter_stride=[1, 1, 1, 1], filter_padding="SAME",name ="conv53")
# Relu_53
layer_conv5 = layers.new_relu_layer(layer_conv5, name="relu53")
# Pooling_5
layer_conv5 = layers.new_pool_layer(input_tensor=layer_conv5, 
ker_size=[1, 2, 2, 1], ker_stride=[1, 2, 2, 1], ker_padding="VALID",name="pool4")
print(layer_conv5)

# Flatten Layer
num_features = layer_conv5.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_conv5, [-1, num_features])
print(layer_flat)

# Fully-Connected Layer 1
layer_fc1 = layers.new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=4096, name="fc1")
# RelU layer 4
layer_relu6 = layers.new_relu_layer(layer_fc1, name="relu6")
print(layer_relu6)

# Fully-Connected Layer 2
layer_fc2 = layers.new_fc_layer(layer_relu6, num_inputs=4096, num_outputs=4096, name="fc2")
# RelU layer 4
layer_relu7 = layers.new_relu_layer(layer_fc2, name="relu7")
print(layer_relu7)

# Fully-Connected Layer 3
layer_output = layers.new_fc_layer(input=layer_relu7, num_inputs=4096, num_outputs=image_types, name="fc3")
print(layer_output)

# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_output)

# Use Cross entropy cost function
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_output, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_true, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(model_path_name + "/train")
writer_valid = tf.summary.FileWriter(model_path_name + "/valid")
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
        start_time = time.time()
        train_accuracy = 0
        num_batch = int(dataset.get_num_train()/BATCH_SIZE + 1)
        print("training epoch ", epoch, "num batch ", num_batch, " size batch ", BATCH_SIZE)
        for batch in range(0, num_batch):
            # Get a batch of images and labels
            dataset.load_data_train_next(BATCH_SIZE)
            x_batch, y_batch = dataset.x_train, dataset.y_train
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
        print("model saved:", os.path.join(model_path_name, "model.ckpt"))
        train_accuracy /= num_batch

        # Generate summary and validate the model on the entire validation set
        print("validating epoch ", epoch)
        vali_accuracy = 0
        num_test_patch = 0.001
        while dataset.load_data_test_next(BATCH_SIZE):
            summ, vali_accuracy_temp = sess.run([merged_summary, accuracy], feed_dict={x_train:dataset.x_test, y_true:dataset.y_test})
            writer_valid.add_summary(summ, valid_num_loop)
            vali_accuracy += vali_accuracy_temp
            num_test_patch += 1.0
            valid_num_loop += 1
        vali_accuracy /= num_test_patch
        
        end_time = time.time()

        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
