import numpy as np
import cv2
import os


def crop_center(img, w, h):
    crop_size = min(img.shape[0:2])
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(crop_size//2)
    starty = y//2-(crop_size//2)
    crop_img = img[starty:starty+crop_size, startx:startx+crop_size]
    resize_img = cv2.resize(crop_img, (h, w)) 
    return resize_img

def standerlize_size(root_folder):
    list_folder = os.listdir(root_folder)
    list_image = os.listdir(root_folder)
    for name_image in list_image:
        image_name = root_folder  + "/" + name_image
        image = cv2.imread(image_name)
        if image.shape[0] != 480 or image.shape[1] != 640 or image.shape[2] != 3:
            print(image.shape)
            image = crop_center(image, 480, 640)
            print("crop_center: ", image_name)
            cv2.imwrite(image_name, image)

if __name__ == "__main__":
    standerlize_size("media")

