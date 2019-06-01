import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import json
import matplotlib.pyplot as plt
import network_config as config

np.set_printoptions(precision=4)

model_path_name= "bknet_model"

# image param
with open(config.lable_file_name) as json_file:    
            dict_lables = json.load(json_file)
            print("predict parse dict_lables ++++++++++++")
            print(dict_lables)
            print("predict parse dict_lables ------------")

image_width = config.image_width
image_height = config.image_height
image_channel = config.image_channel

# input data to predict
def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

image_name = 'D:/Temp/chuamotcot.jpg'
img = cv.imread(image_name)
crop_size = min(img.shape[0:2])
crop_img = crop_center(img, crop_size, crop_size)
resize_img = cv.resize(crop_img, (image_width, image_height))

# show image
plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(crop_img)
plt.figure(3)
plt.imshow(resize_img)
plt.show()

predict_img = resize_img.reshape(1, image_width, image_height, image_channel)

labels = np.zeros((1, dict_lables["num_type"]))

# begin session
session = tf.Session()

# create a saver object to load the model
saver = tf.train.import_meta_graph(os.path.join(model_path_name, 'model.ckpt.meta'))

# restore the model from our checkpoints folder
saver.restore(session, os.path.join(model_path_name, 'model.ckpt'))

# create graph object for getting the same network architecture
graph = tf.get_default_graph()

# get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("fc2/add:0")

# create placeholders to pass the image and get output labels
im_ph = graph.get_tensor_by_name("x_train:0")
label_ph = graph.get_tensor_by_name("y_true:0")

# inorder to make the output to be either 0 or 1.
network = tf.nn.softmax(network)

# creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {im_ph: predict_img, label_ph: labels}

result = session.run(network, feed_dict=feed_dict_testing)

# show data
print(result)
print(dict_lables[str(np.argmax(result[0]))])

