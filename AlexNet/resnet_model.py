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
model_path_name= "resnet_model"
#model_path_name= config.root_path + "/code_project/resnet_model" #for google colab

# Config trainning
NUM_EPOCHS = 50
BATCH_SIZE = 128

# Model parameters
image_width = config.image_width
image_height = config.image_height
image_channel = config.image_channel
image_types = dataset.get_num_types()

# Placeholder variable for the input images
x_train = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='x_train')
# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, image_types], name='y_true')

tensor = layers.new_conv_bn_relu_layer(x_train, [7, 7, 3, 64], 2, "conv10")
print(tensor)

tensor = tf.nn.max_pool(value = tensor, ksize = [1, 2, 2, 1], strides= [1, 2, 2, 1], padding="SAME", name="max10")
print(tensor)

tensor = layers.new_residual_block(input_layer=tensor, output_channel = 64, first_block=False, name="block20")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 64, first_block=False, name="block21")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 64, first_block=False, name="block22")
print(tensor)

tensor = layers.new_residual_block(input_layer=tensor, output_channel = 128, first_block=True, name="block30")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 128, first_block=False, name="block31")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 128, first_block=False, name="block32")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 128, first_block=False, name="block33")
print(tensor)

tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=True, name="block40")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=False, name="block41")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=False, name="block42")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=False, name="block43")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=False, name="block44")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 256, first_block=False, name="block45")
print(tensor)

tensor = layers.new_residual_block(input_layer=tensor, output_channel = 512, first_block=True, name="block50")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 512, first_block=False, name="block51")
tensor = layers.new_residual_block(input_layer=tensor, output_channel = 512, first_block=False, name="block52")
print(tensor)

tensor = tf.nn.avg_pool(value = tensor, ksize = [1, 2, 2, 1], strides= [1, 2, 2, 1], padding="SAME", name="avg50")
print(tensor)

len_tensor = tensor.get_shape()[1:4].num_elements()
tensor = tf.reshape(tensor, [-1, len_tensor])
print(tensor)

# Fully Connected Layer 1 | Dropout
tensor = layers.new_fc_layer(input=tensor, num_inputs=len_tensor, num_outputs= 4096, name="fc_layer1")
print(tensor)

# Fully Connected Layer 2
layer_output = layers.new_fc_layer(input=tensor, num_inputs=4096, num_outputs= image_types, name="fc_layer2")
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
    saver.save(sess, os.path.join(model_path_name, "model.ckpt"))
    exit()
    # Loop over number of epochs
    for epoch in range(NUM_EPOCHS):
        epoch_test = int(dataset.get_num_test() / BATCH_SIZE)
        train_intev = int(dataset.get_num_train() / BATCH_SIZE / epoch_test)
        epoch_train = 0
        print("num_train=", dataset.get_num_train(), " num_test=", dataset.get_num_test(), " train_intev=", train_intev)
        start_time = time.time()
        train_accuracy = 0
        num_batch = int(dataset.get_num_train()/BATCH_SIZE)
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
            if epoch_train % train_intev == 0:
                summ = sess.run(merged_summary, feed_dict=feed_dict_train)
                writer_train.add_summary(summ, train_num_loop)
                train_num_loop += 1
            epoch_train += 1

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
        print("\t- Training Accuracy:\t{}".format(train_accuracy))
        print("\t- Validation Accuracy:\t{}".format(vali_accuracy))
        print("")