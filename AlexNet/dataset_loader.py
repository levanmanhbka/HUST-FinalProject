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
        self.load_datasets_random(config.training_buffer)
        self.x_test_index = 0
        self.x_test_max = self.get_num_test()

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

    def load_datasets_random(self, num_image):
        x_train = []
        y_train = []
        if num_image > self.dict_lables["num_train"]:
            num_image = self.dict_lables["num_train"]
        list_samples = os.listdir(config.train_folder_path)
        list_chosed = random.sample(list_samples, int(num_image))
        for sample_name in list_chosed:
            x,y = load_sample(config.train_folder_path, sample_name)
            x_train.append(x)
            y_train.append(y)
        # one hot encoding    
        y_train = make_one_host(y_train, self.get_num_types())
        # convert to numpy array
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def load_data_test_continuos(self):
        x_test = []
        y_test = []
        # load data test
        testing_samples = os.listdir(config.test_folder_path)
        num_sample = self.x_test_max - self.x_test_index
        if num_sample > config.testing_buffer:
            num_sample = config.testing_buffer
        elif num_sample <= 1:
            return False
        for index in range(self.x_test_index, self.x_test_index + num_sample):
            x, y = load_sample(config.test_folder_path, testing_samples[index])
            x_test.append(x)
            y_test.append(y)
        # make one hot data
        y_test = make_one_host(y_test, self.get_num_types())
        # convert to numpy array
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.x_test_index += num_sample
        return True
    
    def reset_data_test_index(self):
        self.x_test_index = 0

    def __batch_from_loaded(self, batch_size):
        x_train = []
        y_train = []
        train_len = len(self.y_train)
        select_list = random.sample(range(train_len), batch_size)
        for select in select_list:
            x_train.append(self.x_train[select])
            y_train.append(self.y_train[select])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    def load_batch_dataset(self, batch_size):
        return self.__batch_from_loaded(batch_size)

    def get_num_types(self):
        return self.dict_lables["num_type"]
    
    def get_num_train(self):
        num_train = self.y_train.shape[0]
        return num_train
    
    def get_num_test(self):
        return self.dict_lables["num_test"]
        
if __name__ == "__main__":
    data_loader = DatasetLoader()