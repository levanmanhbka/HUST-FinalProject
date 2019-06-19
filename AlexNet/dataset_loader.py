import os
import shutil
import numpy as np
import cv2 as cv
import random
import json

import network_config as config

def load_sample(file_path, file_name):
    y = int(file_name.split('_')[0])
    x = cv.imread(file_path + "/" +file_name)
    return x, y

def make_one_host(m_list, w):
    # w = max(m_list) + 1
    h = len(m_list)
    a = np.array(m_list, dtype = int)
    b = np.zeros((h, w), dtype = float)
    b[np.arange(h), a] = 1
    return b

class DatasetLoader():
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    dict_lables = None

    def __init__(self):
        with open(config.lable_file_name) as json_file:    
            self.dict_lables = json.load(json_file)
            print("DatasetLoader dict_lables ++++++++++++")
            print(self.dict_lables)
            print("DatasetLoader dict_lables ------------")
        self.x_train_index = 0
        self.list_train_image_name = os.listdir(config.train_folder_path)
        self.list_train_image_name = random.sample(self.list_train_image_name, len(self.list_train_image_name))
        self.x_test_index = 0
        self.list_test_image_name = os.listdir(config.test_folder_path)
        self.list_test_image_name = random.sample(self.list_test_image_name, len(self.list_test_image_name))

    def __load_all_datasets(self):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        # load data train
        training_samples = os.listdir(config.train_folder_path)
        for sample_name in training_samples:
            x,y = load_sample(config.train_folder_path, sample_name)
            x_train.append(x)
            y_train.append(y)
        # load data test
        testing_samples = os.listdir(config.test_folder_path)
        for sample_name in testing_samples:
            x, y = load_sample(config.test_folder_path, sample_name)
            x_test.append(x)
            y_test.append(y)
        # one hot encoding    
        y_train = make_one_host(y_train, self.get_num_types())
        y_test = make_one_host(y_test, self.get_num_types())
        # convert to numpy array
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def augmentation(self, image):
        rows, cols, depth = image.shape
        mode = np.random.choice(range(0, 3))
        #print("mode", mode)
        if mode == 1:
            angle = np.random.choice(range(-20, 20))
            M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
            image = cv.warpAffine(image,M,(cols,rows))
            #print("angle", angle)
        elif mode == 2:
            ychange = np.random.choice(range(-40, 40))
            xchange = np.random.choice(range(-40, 40))
            M = np.float32([[1,0,ychange],[0,1,xchange]])
            image = cv.warpAffine(image,M,(cols,rows))
            #print(ychange," " ,xchange)
        image = cv.resize(image, (config.image_width, config.image_height))
        return image

    def load_data_train_next(self, batch_size):
        x_train = []
        y_train = []
        if len(self.list_train_image_name) < batch_size:
            batch_size = len(self.list_train_image_name)
        if self.x_train_index + batch_size > len(self.list_train_image_name):
            self.list_train_image_name = random.sample(self.list_train_image_name, len(self.list_train_image_name))
            self.x_train_index = 0
        list_chosed = self.list_train_image_name[self.x_train_index:self.x_train_index + batch_size]
        for sample_name in list_chosed:
            x,y = load_sample(config.train_folder_path, sample_name)
            x = self.augmentation(x)
            x_train.append(x)
            y_train.append(y)
        # one hot encoding    
        y_train = make_one_host(y_train, self.get_num_types())
        # convert to numpy array
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_train = self.x_train / 255.0
        self.x_train_index += batch_size
        return True

    def load_data_test_next(self, batch_size):
        x_test = []
        y_test = []
        # load data test
        if len(self.list_test_image_name) < batch_size:
            batch_size = len(list_test_image_name)
        if self.x_test_index + batch_size > len(self.list_test_image_name):
            self.list_test_image_name = random.sample(self.list_test_image_name, len(self.list_test_image_name))
            self.x_test_index = 0
            return False
        list_chosed = self.list_test_image_name[self.x_test_index:self.x_test_index + batch_size]
        for sample_name in list_chosed:
            x, y = load_sample(config.test_folder_path, sample_name)
            x_test.append(x)
            y_test.append(y)
        # make one hot data
        y_test = make_one_host(y_test, self.get_num_types())
        # convert to numpy array
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.x_test = self.x_test / 255.0
        self.x_test_index += batch_size
        return True
    
    def get_num_types(self):
        return self.dict_lables["num_type"]
    
    def get_num_train(self):
        num_train = self.dict_lables["num_train"]
        return num_train
    
    def get_num_test(self):
        return self.dict_lables["num_test"]
        
if __name__ == "__main__":
    data_loader = DatasetLoader()