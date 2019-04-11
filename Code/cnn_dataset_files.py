import os
import shutil
import numpy as np
import cv2 as cv
import random


train_folder_path = "train_folder"
test_folder_path = "test_folder"

#create trainfolder
def make_dataset_folder():
    if not os.path.exists(train_folder_path):
        os.mkdir(train_folder_path)
    if not os.path.exists(test_folder_path):
        os.mkdir(test_folder_path)

 #3 part of samples to train, 1 part of samples to test
def copy_files_train(src_folder, id_class):
    src_list_file = os.listdir(src_folder)
    num_sample = len(src_list_file)
    num_test = int(num_sample / 4)
    list_test_select = random.sample(range(num_sample), num_test)
    list_test_select.sort()

    file_number = 0
    test_number = 0
    train_number = 0
    test_select = 0

    for file_name in src_list_file:
        src_file = src_folder + "/" + file_name
        dst_file = ""
        if (file_number == list_test_select[test_select]):
            dst_file = test_folder_path + "/" + str(id_class) + "_" + str(test_number) + ".jpg"
            if test_select < num_test - 1:
                test_select = test_select + 1
            test_number = test_number + 1
        else:
            dst_file = train_folder_path + "/" + str(id_class) + "_" + str(train_number) + ".jpg"
            train_number = train_number + 1

        shutil.copy(src_file, dst_file)
        file_number = file_number + 1
        
    return file_number, test_number, train_number

def separate_sample():
    file_number0, test_number0, train_number0 = copy_files_train("flowers/daisy", 0)
    file_number1, test_number1, train_number1 = copy_files_train("flowers/dandelion", 1)
    file_number2, test_number2, train_number2 = copy_files_train("flowers/rose", 2)
    file_number3, test_number3, train_number3 = copy_files_train("flowers/tulip", 3)
    file_number4, test_number4, train_number4 = copy_files_train("flowers/sunflower", 4)

def load_sample(file_path, file_name):
    y = int(file_name.split('_')[0])
    x = cv.imread(file_path + "/" +file_name)
    x = cv.resize(x, (128, 128))
    return x, y

def make_one_host(m_list):
    w = max(m_list) + 1
    h = len(m_list)
    a = np.array(m_list, dtype = int)
    b = np.zeros((h, w), dtype = float)
    b[np.arange(h), a] = 1
    return b

def load_datasets():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    training_samples = os.listdir(train_folder_path)
    for sample_name in training_samples:
        x,y = load_sample(train_folder_path, sample_name)
        X_train.append(x)
        y_train.append(y)
    
    testing_samples = os.listdir(test_folder_path)
    for sample_name in testing_samples:
        x, y = load_sample(test_folder_path, sample_name)
        X_test.append(x)
        y_test.append(y)
    y_train = make_one_host(y_train)
    y_test = make_one_host(y_test)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return  X_train, y_train, X_test, y_test