import numpy as np
import cv2
import os


def crop_center(img, size_img):
    status = True
    if img.shape[2] != 3:
        status = Flase
    crop_size = min(img.shape[0:2])
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(crop_size//2)
    starty = y//2-(crop_size//2)
    crop_img = img[starty:starty+crop_size, startx:startx+crop_size]
    resize_img = cv2.resize(crop_img, (size_img, size_img)) 
    return resize_img

def standerlize_size(root_folder):
    list_folder = os.listdir(root_folder)
    for folder in list_folder:
        list_image = os.listdir(root_folder+ "/" + folder)
        print("working on:" + root_folder+ "/" + folder +"------>")
        image_number = 1
        for name_image in list_image:
            image_name = root_folder + "/" + folder + "/" + name_image
            image = cv2.imread(image_name)
            if image is None:
                os.remove(image_name)
                print("remove: " + image_name)
                continue
            elif image.shape[0] != 480 or image.shape[1] != 480 or image.shape[2] != 3:
                print(image.shape)
                image = crop_center(image, 480)
                print("crop_center: ", image_name)
                #cv2.imshow("image", image)
                cv2.imwrite(image_name, image)
                cv2.waitKey(30)
            os.rename(image_name, root_folder + "/" + folder + "/im_"+str(image_number) + ".jpg")
            image_number += 1 

def change_folder_name(root_folder):
    list_folder = os.listdir(root_folder)
    folder_count = 1
    for folder in list_folder:
        sub2 = folder.split("_")[1]
        os.rename(root_folder + "/" + folder, root_folder + "/" + str(folder_count) + "_" + sub2)
        folder_count += 1

def extract_image():
    cap = cv2.VideoCapture("../../HoaCamTuCau.mp4")
    count = 0
    img_number = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        #frame = crop_center(frame, 480)
        #cv2.imshow('frame',frame)
        if count % 90 ==0:
            frame = crop_center(frame, 480)
            cv2.imwrite("D:/Temp/camtucau_"+str(img_number)+".jpg", frame)
            img_number += 1
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    standerlize_size("folder_dectected")

