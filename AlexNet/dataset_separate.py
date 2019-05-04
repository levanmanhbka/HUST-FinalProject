import os
import shutil
import numpy as np
import cv2 as cv
import random
import json

import network_config as config

#1 create trainfolder
def create_dataset_folder():
    if not os.path.exists(config.train_folder_path):
        os.mkdir(config.train_folder_path)
    if not os.path.exists(config.test_folder_path):
        os.mkdir(config.test_folder_path)

 #3 separate folder part of samples to train, 1 part of samples to test
def separate_folder(src_folder, id_class):
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
            dst_file = config.test_folder_path + "/" + str(id_class) + "_" + str(test_number) + ".jpg"
            if test_select < num_test - 1:
                test_select = test_select + 1
            test_number = test_number + 1
        else:
            dst_file = config.train_folder_path + "/" + str(id_class) + "_" + str(train_number) + ".jpg"
            train_number = train_number + 1

        image = cv.imread(src_file)
        if image is None:
            os.remove(src_file)
        elif len(image.shape) == 3:
            image = cv.resize(image, (config.image_width, config.image_height))
            cv.imwrite(dst_file, image)
            file_number = file_number + 1
        else:
            os.remove(src_file)
        
        
    return file_number, test_number, train_number

#2 separate all folder
def separate_all_folder():
    list_folder_name = os.listdir(config.input_folder_path)
    floder_index = 0
    dict_lables = {}
    file_number, test_number, train_number = 0, 0, 0
    for folder_name in list_folder_name:
        print("Loading ", floder_index, " / ", len(list_folder_name))
        n0, n1, n2 = separate_folder(config.input_folder_path + "/" + folder_name, floder_index)
        file_number += n0
        test_number += n1
        train_number += n2
        # write lable dictionary
        dict_lables[str(floder_index)] = folder_name
        floder_index += 1
    # write dataset info to dictionary    
    dict_lables["num_file"]= file_number
    dict_lables["num_test"]= test_number
    dict_lables["num_train"]= train_number
    dict_lables["num_type"]= floder_index
    # write dictionary to file
    with open(config.lable_file_name, 'w') as json_file:
        json.dump(dict_lables, json_file)

if __name__ == '__main__':
    create_dataset_folder()
    separate_all_folder()