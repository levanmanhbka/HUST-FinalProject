import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from home.network_predic import predict_image_function

class ReoderLanmark():

    def standed_image(self, image):
        # img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = image
        return img

    def get_new_order(self, anchor_name, num_lanmark=64):
        print("get_new_order: anchor= ", anchor_name)
        
        predict_result = predict_image_function(anchor_name)
        print("predict_result: ", predict_result)

        image = cv.imread(anchor_name)
        image = self.standed_image(image=image)
        cv.imshow("image", image)
        cv.waitKey(2000)
        cv.destroyAllWindows()
        return predict_result