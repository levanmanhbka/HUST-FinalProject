import tensorflow as tf
import time
from dataset_loader import DatasetLoader

# General parameters of the model
NUM_EPOCHS = 50
BATCH_SIZE = 128
DROPOUT_KEEP_PROB = 0.5
K_BIAS = 2
N_DEPTH_RADIUS = 5
ALPHA = 1e-4
BETA = 0.75

# Global dataset dictionary
dataset_dict = {
    "image_size": 228,
    "num_channels": 3,
    "num_labels": 4,
}

# Filter shapes for each layer 
conv_filter_shapes = {
    "c1_filter": [11, 11, 3, 96],
    "c2_filter": [5, 5, 96, 256],
    "c3_filter": [3, 3, 256, 384],
    "c4_filter": [3, 3, 384, 384],
    "c5_filter": [3, 3, 384, 256]
}

# Fully connected shapes
fc_connection_shapes = {
    "f1_shape": [6*6*256, 4096],
    "f2_shape": [4096, 4096],
    "f3_shape": [4096, dataset_dict["num_labels"]]
}

# Weights for each layer
conv_weights = {
    "c1_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c1_filter"], stddev=0.05, dtype=tf.float32), name="c1_weights"),
    "c2_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c2_filter"], stddev=0.05, dtype=tf.float32), name="c2_weights"),
    "c3_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c3_filter"], stddev=0.05, dtype=tf.float32), name="c3_weights"),
    "c4_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c4_filter"], stddev=0.05, dtype=tf.float32), name="c4_weights"),
    "c5_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c5_filter"], stddev=0.05, dtype=tf.float32), name="c5_weights"),
    "f1_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f1_shape"], stddev=0.05, dtype=tf.float32), name="f1_weights"),
    "f2_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f2_shape"], stddev=0.05, dtype=tf.float32), name="f2_weights"),
    "f3_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f3_shape"], stddev=0.05, dtype=tf.float32), name="f3_weights")
}

# Biases for each layer
conv_biases = {
    "c1_biases": tf.Variable(tf.truncated_normal([conv_filter_shapes["c1_filter"][3]], dtype=tf.float32), name="c1_biases"),
    "c2_biases": tf.Variable(tf.truncated_normal([conv_filter_shapes["c2_filter"][3]], dtype=tf.float32), name="c2_biases"), 
    "c3_biases": tf.Variable(tf.truncated_normal([conv_filter_shapes["c3_filter"][3]], dtype=tf.float32), name="c3_biases"),
    "c4_biases": tf.Variable(tf.truncated_normal([conv_filter_shapes["c4_filter"][3]], dtype=tf.float32), name="c4_biases"),
    "c5_biases": tf.Variable(tf.truncated_normal([conv_filter_shapes["c5_filter"][3]], dtype=tf.float32), name="c5_biases"),
    "f1_biases": tf.Variable(tf.truncated_normal([fc_connection_shapes["f1_shape"][1]], stddev=0.05, dtype=tf.float32), name="f1_biases"),
    "f2_biases": tf.Variable(tf.truncated_normal([fc_connection_shapes["f2_shape"][1]], stddev=0.05, dtype=tf.float32), name="f2_biases"),
    "f3_biases": tf.Variable(tf.truncated_normal([fc_connection_shapes["f3_shape"][1]], stddev=0.05, dtype=tf.float32), name="f3_biases")
}

# Declare the input and output placeholders
input_img = tf.placeholder(tf.float32, shape=[None, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
labels = tf.placeholder(tf.float32, shape=[None, dataset_dict["num_labels"]])

# Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
c_layer_1 = tf.nn.conv2d(input_img, conv_weights["c1_weights"], strides=[1, 4, 4, 1], padding="VALID", name="c_layer_1")
c_layer_1 += conv_biases["c1_biases"]
c_layer_1 = tf.nn.relu(c_layer_1)
c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name = "max_pool_1")
print(c_layer_1)

# Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
c_layer_2 = tf.nn.conv2d(c_layer_1, conv_weights["c2_weights"], strides=[1, 1, 1, 1], padding="VALID", name="c_layer_2")
c_layer_2 += conv_biases["c2_biases"]
c_layer_2 = tf.nn.relu(c_layer_2)
c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=5, bias=K_BIAS, alpha=ALPHA, beta=BETA)
c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name = "max_pool_2")
print(c_layer_2)

# Convolution Layer 3 | ReLU
c_layer_3 = tf.nn.conv2d(c_layer_2, conv_weights["c3_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
c_layer_3 += conv_biases["c3_biases"]
c_layer_3 = tf.nn.relu(c_layer_3)
print(c_layer_3)

# Convolution Layer 4 | ReLU
c_layer_4 = tf.nn.conv2d(c_layer_3, conv_weights["c4_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
c_layer_4 += conv_biases["c4_biases"]
c_layer_4 = tf.nn.relu(c_layer_4)
print(c_layer_4)

# Convolution Layer 5 | ReLU | Max Pooling
c_layer_5 = tf.nn.conv2d(c_layer_4, conv_weights["c5_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
c_layer_5 += conv_biases["c5_biases"]
c_layer_5 = tf.nn.relu(c_layer_5)
c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name = "max_pool_5")
print(c_layer_5)

# Flatten the multi-dimensional outputs to feed fully connected layers
feature_map = tf.reshape(c_layer_5, [-1, 6 * 6 * 256])
print(feature_map)

# Fully Connected Layer 1 | Dropout
fc_layer_1 = tf.matmul(feature_map, conv_weights["f1_weights"]) + conv_biases["f1_biases"]
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=DROPOUT_KEEP_PROB)
print(fc_layer_1)

# Fully Connected Layer 2 | Dropout
fc_layer_2 = tf.matmul(fc_layer_1, conv_weights["f2_weights"]) + conv_biases["f2_biases"]
fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=DROPOUT_KEEP_PROB)
print(fc_layer_2)

# Fully Connected Layer 3 | Softmax
fc_layer_3 = tf.matmul(fc_layer_2, conv_weights["f3_weights"]) + conv_biases["f3_biases"]
cnn_output = tf.nn.softmax(fc_layer_3)
print(cnn_output)


#--------------------------- Training model -------------------------#

y_pred = cnn_output

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc_layer_3, labels= labels))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Datasets
dataset  = DatasetLoader()

with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # Loop over number of eporchs
    train_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for batch in range(0, int(dataset.get_num_train()/BATCH_SIZE)):
            # Get a batch of images and labels
            x_batch, y_batch = dataset.load_batch_dataset(BATCH_SIZE)
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {input_img: x_batch, labels: y_batch}
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
        train_accuracy = train_accuracy / (dataset.get_num_train()/BATCH_SIZE)
        # Validate the model on the entire validation set
        vali_accuracy = sess.run(accuracy, feed_dict={input_img:dataset.x_test, labels:dataset.y_test})
        end_time = time.time()
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
