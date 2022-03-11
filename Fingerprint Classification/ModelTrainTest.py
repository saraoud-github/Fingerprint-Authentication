import os, shutil
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import random

#Get all the files in the current working directory
cwd = os.listdir(os.getcwd())
DATASET_TEXT_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\TextFiles")
DATASET_PNG_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\PNGFiles")
CLASSES_PATH = os.path.join(cwd[0], "Classes")

#Create classes list
categories = ['A', 'L', 'R', 'T', 'W']

#Define kernel for dilation (morphological operation)
kernel= np.ones((5,5), np.uint8)

# # #Create a folder for each class
# # for category in categories:
# #     os.makedirs(os.path.join(CLASSES_PATH,category))
# #
# # #Remove last 3 characters from PNG images in dataset
# # for PNG_file in os.listdir(DATASET_PNG_PATH):
# #     os.rename(os.path.join(DATASET_PNG_PATH,PNG_file), os.path.join(DATASET_PNG_PATH,PNG_file.replace(PNG_file[5:8:1], '')))
#
# #Label PNG images according to info in TXT files
# # for TXT_file in os.listdir(DATASET_TEXT_PATH):
# #         fileObject = open(os.path.join(DATASET_TEXT_PATH, TXT_file), 'r')
# #         for i, line in enumerate(fileObject):
# #             if i==1:
# #                 class_str = line[7]
# #             if i==2:
# #                 img = line[9:14:1]
# #                 shutil.copy(os.path.join(DATASET_PNG_PATH, img+'.png'), os.path.join(CLASSES_PATH, class_str))


####################################################################
##        READ TRAINING IMAGES AND EXTRACT FEATURES
####################################################################
#Define list to hold Gabor features
# gabor_feat = []
# #Used to track which image is being processed
# image_count=1
# #Image resize scale
# SIZE = 128
#
# #Loop through images in each class in categories list
# for category in categories:
#     img_path = os.path.join(CLASSES_PATH, category)
#     label = categories.index(category)
#
#     #Iterate through each image
#     for image in os.listdir(img_path):
#
#         #Temporary data frame to capture information for each loop. Resets dataframe to blank after each loop.
#         df = pd.DataFrame(dtype='float16')
#
#         #Read each image
#         input_img = cv2.imread(os.path.join(img_path,image))
#         #Resize image
#         input_img = cv2.resize(input_img, (SIZE, SIZE))
#         #Apply morphological operations (erosion, dilation, and opening) to each image
#         # img_erosion = cv2.erode(img, kernel, iterations=1)
#         # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
#         # img_opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel)
#         # cv2.imshow('Input', fingerprint_img)
#         # cv2.imshow('Erosion', img_erosion)
#         # cv2.imshow('Dilation', img_dilation)
#         # cv2.imshow('Opening', img_opening)
#         # cv2.waitKey(0)
#
#         # Check if the input image is RGB or grey and convert to grey if RGB
#         if input_img.ndim == 3 and input_img.shape[-1] == 3:
#             img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#         elif input_img.ndim == 2:
#             img = input_img
#         else:
#             raise Exception("The module works only with grayscale and RGB images!")
#
#         ################################################################
#         # START ADDING DATA TO THE DATAFRAME
#         ################################################################
#
#         # Add pixel values to the data frame
#         pixel_values = img.reshape(-1)
#         df['Pixel_Value'] = pixel_values  # Pixel value itself as a feature
#
#         ############################################################################
#         # Generate Gabor features
#         #Track number of Gabor filter generated
#         num = 1
#         kernels = []
#         for theta in range(2):  # Define number of thetas
#             theta = theta / 4. * np.pi
#             for sigma in (1, 3):  # Sigma with 1 and 3 and 5
#                 for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
#                     for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
#
#                         gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
#                         #print(gabor_label)
#                         ksize = 9
#                         kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
#                         kernels.append(kernel)
#                         #Apply Gabor filter to the image and add values to a new column
#                         fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#                         filtered_img = fimg.reshape(-1)
#                         df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
#                         #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
#                         num += 1  # Increment for gabor column label
#
#
#         #print(df.info(verbose=True, memory_usage='deep'))
#         ######################################
#         # Add generated Gabor features, label, and image name of each corresponding processed image
#         gabor_feat.append([df, label, image])
#         print(image_count)
#         image_count +=1
#
# print('Done!')
#
# # Assign training features (Gabor features) to X and labels to Y
#
# #Write the data into a pickle file before training
# pick_in = open('data_Gabor.pickle', 'wb')
# pickle.dump(gabor_feat, pick_in)
# pick_in.close()

#Read the pickle file containing the labeled data
pick_in = open('data_Gabor.pickle', 'rb')
#Load the pickle file into data variable
gabor_data = pickle.load(pick_in)
pick_in.close()
print('Done reading pickle file')

#Shuffle the data
random.shuffle(gabor_data)
gabor_features = []
labels = []
image_names = []
#Split the elements in data into features and labels
for feature, label, img_name in gabor_data:
    gabor_features.append([np.array(feature).flatten()])
    labels.append([label])
    image_names.append(img_name)

print('Done splitting features from labels')


#Split the data into train (70%) and test data (30%)
x_train, x_test, y_train, y_test = train_test_split(gabor_features, labels, test_size=0.2)


####################################################################
##STEP 4: Define the classifier and fit a model with our training data
####################################################################

# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model on training data
nsamples, nx, ny = np.array(x_train).shape
x_train_1 = np.array(x_train).reshape((nsamples, nx*ny))
print('Started training')
model.fit(x_train_1, np.ravel(y_train))
print('Done training')

# #Saves the model in 'model.sav' folder
# pick = open('RFmodel.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

#######################################################
# STEP 5: Accuracy check
#########################################################
#  #Opens and reads the model
# pick = open('RFmodel.sav', 'rb')
# model = pickle.load(pick)
# pick.close()

nsamples, nx, ny = np.array(x_test).shape
x_test_1 = np.array(x_test).reshape((nsamples, nx*ny))
prediction_test = model.predict(x_test_1)
##Check accuracy on test dataset.
print("Accuracy = ", metrics.accuracy_score(np.ravel(y_test), prediction_test))

# from yellowbrick.classifier import ROCAUC
#
# print("Classes in the image are: ", np.unique(Y))
#
# # ROC curve for RF
# roc_auc = ROCAUC(model, classes=[0, 1, 2, 3])  # Create object
# roc_auc.fit(X_train, y_train)
# roc_auc.score(X_test, y_test)
# roc_auc.show()

##########################################################
# STEP 6: SAVE MODEL FOR FUTURE USE
###########################################################
##You can store the model for future use. In fact, this is how you do machine elarning
##Train on training images, validate on test images and deploy the model on unknown images.
#
#
##Save the trained model as pickle string to disk for future use

pick = open('RFmodel.sav', 'wb')
pickle.dump(model, pick)
pick.close()

# model_name = "sandstone_model_multi_image"
# pickle.dump(model, open(model_name, 'wb'))
#
##To test the model on future datasets
# loaded_model = pickle.load(open(model_name, 'rb'))



