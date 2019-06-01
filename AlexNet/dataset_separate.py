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
    list_src_image = os.listdir(src_folder)
    file_number = len(list_src_image)
    test_number = int(file_number / 4)
    train_number = file_number - test_number

    list_src_image = random.sample(list_src_image, file_number)
    list_train_image = list_src_image[test_number:file_number]
    list_test_image = list_src_image[0:test_number]
    
    file_number = 0
    train_number = 0
    test_number = 0

    for file_name in list_train_image:
        src_file = src_folder + "/" + file_name
        dst_file = config.train_folder_path + "/" + str(id_class) + "_" + str(train_number) + ".jpg"
        image = cv.imread(src_file)
        if image is None:
            os.remove(src_file)
        elif len(image.shape) == 3:
            image = cv.resize(image, (config.image_width, config.image_height))
            cv.imwrite(dst_file, image)
            train_number = train_number + 1
            file_number = file_number + 1
        else:
            os.remove(src_file)

    for file_name in list_test_image:
        src_file = src_folder + "/" + file_name
        dst_file = config.test_folder_path + "/" + str(id_class) + "_" + str(test_number) + ".jpg"
        image = cv.imread(src_file)
        if image is None:
            os.remove(src_file)
        elif len(image.shape) == 3:
            image = cv.resize(image, (config.image_width, config.image_height))
            cv.imwrite(dst_file, image)
            test_number = test_number + 1
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