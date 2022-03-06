import os, shutil
import cv2
import numpy as np

#Get all the files in the current working directory
cwd = os.listdir(os.getcwd())
DATASET_TEXT_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\TextFiles")
DATASET_PNG_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\PNGFiles")
CLASSES_PATH = os.path.join(cwd[0], "Classes")

#Create classes list
categories = ['A', 'L', 'R', 'T', 'W']

#Define kernel for dilation (morphological operation)
kernel_dil = np.ones((5,5), np.uint8)

# #Create a folder for each class
# for category in categories:
#     os.makedirs(os.path.join(CLASSES_PATH,category))
#
# #Remove last 3 characters from PNG images in dataset
# for PNG_file in os.listdir(DATASET_PNG_PATH):
#     os.rename(os.path.join(DATASET_PNG_PATH,PNG_file), os.path.join(DATASET_PNG_PATH,PNG_file.replace(PNG_file[5:8:1], '')))

#Label PNG images according to info in TXT files
# for TXT_file in os.listdir(DATASET_TEXT_PATH):
#         fileObject = open(os.path.join(DATASET_TEXT_PATH, TXT_file), 'r')
#         for i, line in enumerate(fileObject):
#             if i==1:
#                 class_str = line[7]
#             if i==2:
#                 img = line[9:14:1]
#                 shutil.copy(os.path.join(DATASET_PNG_PATH, img+'.png'), os.path.join(CLASSES_PATH, class_str))

for category in categories:
    path = os.path.join(CLASSES_PATH, category)
    label = categories.index(category)
    for img in os.listdir(path):
        fingerprint_img = cv2.imread(img, 0)
        img_dilation = cv2.dilate(fingerprint_img, kernel_dil, iterations=1)
        cv2.imshow('Input', fingerprint_img)
        cv2.imshow('Dilation', img_dilation)

