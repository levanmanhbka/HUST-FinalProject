import os
import shutil
import numpy as np
import cv2 as cv
import random
import json


input_folder_path = "lanmark"
train_folder_path = "train_folder"
test_folder_path = "test_folder"
lable_file_name = "data_lable.json"

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

        image = cv.imread(src_file)
        if image is None:
            os.remove(src_file)
        elif len(image.shape) == 3:
            shutil.copy(src_file, dst_file)
            file_number = file_number + 1
        else:
            os.remove(src_file)
        
        
    return file_number, test_number, train_number

def separate_sample():
    list_folder_name = os.listdir(input_folder_path)
    file_id = 0
    dict_lables = {}
    for folder in list_folder_name:
        copy_files_train(input_folder_path + "/" + folder, file_id)
        # write lable dictionary
        dict_lables[str(file_id)] = folder
        file_id += 1
        print("Loading ", file_id, " / ", len(list_folder_name))
    with open(lable_file_name, 'w') as json_file:
        json.dump(dict_lables, json_file)


def load_sample(file_path, file_name):
    y = int(file_name.split('_')[0])
    x = cv.imread(file_path + "/" +file_name)
    x = cv.resize(x, (256, 256))
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
    
    return  X_train, y_train, X_test, y_test, len(os.listdir(input_folder_path))
