import os, shutil

#Get all the files in the current working directory
cwd = os.listdir(os.getcwd())
DATASET_TEXT_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\TextFiles")
DATASET_PNG_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\PNGFiles")
CLASSES_PATH = os.path.join(cwd[0], "Classes")

#Create classes list
categories = ['A', 'L', 'R', 'T', 'W']

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
