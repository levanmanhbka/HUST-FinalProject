import cv2
import tensorflow as tf
import os
import numpy as np
import cv2 as cv

model_save_name= "cnn_model"

image='chua_mot_cot.jpg'
img=cv.imread(image)
img=cv.resize(img,(128,128))
img=img.reshape(1,128,128,3)

labels = np.zeros((1, 10))

session=tf.Session()


#Create a saver object to load the model
saver = tf.train.import_meta_graph(os.path.join(model_save_name,'.meta'))

#restore the model from our checkpoints folder
saver.restore(session,os.path.join(model_save_name,'.\\'))

#Create graph object for getting the same network architecture
graph = tf.get_default_graph()

#Get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("fc2/add:0")

#create placeholders to pass the image and get output labels
im_ph= graph.get_tensor_by_name("x_train:0")
label_ph = graph.get_tensor_by_name("y_true:0")

#Inorder to make the output to be either 0 or 1.
network=tf.nn.softmax(network)

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {im_ph: img, label_ph: labels}

result=session.run(network, feed_dict=feed_dict_testing)
print(result)
