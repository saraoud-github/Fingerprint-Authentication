import os, shutil
import cv2
import numpy as np
from PIL import Image
import pandas as pd

#Get all the files in the current working directory
cwd = os.listdir(os.getcwd())
DATASET_TEXT_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\TextFiles")
DATASET_PNG_PATH = os.path.join(cwd[0], "NISTSpecialDatabase4GrayScaleImagesofFIGS\\png_txt\\PNGFiles")
CLASSES_PATH = os.path.join(cwd[0], "Classes")

#Create classes list
categories = ['A', 'L', 'R', 'T', 'W']

#Define kernel for dilation (morphological operation)
kernel= np.ones((5,5), np.uint8)

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


####################################################################
##        READ TRAINING IMAGES AND EXTRACT FEATURES
################################################################
image_dataset = pd.DataFrame(dtype='int8') # Dataframe to capture image features
dataset = []
SIZE = 128

# img = cv2.imread(os.path.join(CLASSES_PATH,'A', 'f0005.png'))
# img = cv2.resize(img, (SIZE, SIZE))
# cv2.imshow('img', img)
# cv2.waitKey(0)

for category in categories:
    img_path = os.path.join(CLASSES_PATH, category)
    label = categories.index(category)

    for image in os.listdir(img_path):  # iterate through each file
        #print(image)

        df = pd.DataFrame(dtype='int8')  # Temporary data frame to capture information for each loop.
        # Reset dataframe to blank after each loop.

        img = cv2.imread(os.path.join(img_path,image))
        img = cv2.resize(img, (SIZE, SIZE))
        img_erosion = cv2.erode(img, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        img_opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('Input', fingerprint_img)
        # cv2.imshow('Erosion', img_erosion)
        # cv2.imshow('Dilation', img_dilation)
        # cv2.imshow('Opening', img_opening)
        # cv2.waitKey(0)

        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        # Add pixel values to the data frame
        pixel_values = img_opening.reshape(-1)
        df['Pixel_Value'] = pixel_values  # Pixel value itself as a feature
        df['Image_Name'] = image  # Capture image name as we read multiple images

        ############################################################################
        # Generate Gabor features
        num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):  # Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma with 1 and 3 and 5
                for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                    for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

                        gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                        #                print(gabor_label)
                        ksize = 9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        kernels.append(kernel)
                        # Now filter the image and add values to a new column
                        fimg = cv2.filter2D(img_opening, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
                        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                        num += 1  # Increment for gabor column label

        ######################################
        # Update dataframe for images to include details for each image in the loop
        image_dataset = image_dataset.append(df)
        dataset.append([image_dataset, label])

print('Done!')



# # Assign training features to X and labels to Y
# # Drop columns that are not relevant for training (non-features)
# X = dataset.drop(labels=["Image_Name", "Mask_Name", "Label_Value"], axis=1)
#
# # Assign label values to Y (our prediction)
# Y = dataset["Label_Value"].values
#
# # Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
# from sklearn.preprocessing import LabelEncoder
#
# Y = LabelEncoder().fit_transform(Y)
#
# ##Split data into train and test to verify accuracy after fitting the model.
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
#
# ####################################################################
# # STEP 4: Define the classifier and fit a model with our training data
# ###################################################################
#
# # Import training classifier
# from sklearn.ensemble import RandomForestClassifier
#
# ## Instantiate model with n number of decision trees
# model = RandomForestClassifier(n_estimators=50, random_state=42)
#
# ## Train the model on training data
# model.fit(X_train, y_train)
#
# #######################################################
# # STEP 5: Accuracy check
# #########################################################
#
# from sklearn import metrics
#
# prediction_test = model.predict(X_test)
# ##Check accuracy on test dataset.
# print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#
# from yellowbrick.classifier import ROCAUC
#
# print("Classes in the image are: ", np.unique(Y))
#
# # ROC curve for RF
# roc_auc = ROCAUC(model, classes=[0, 1, 2, 3])  # Create object
# roc_auc.fit(X_train, y_train)
# roc_auc.score(X_test, y_test)
# roc_auc.show()
#
# ##########################################################
# # STEP 6: SAVE MODEL FOR FUTURE USE
# ###########################################################
# ##You can store the model for future use. In fact, this is how you do machine elarning
# ##Train on training images, validate on test images and deploy the model on unknown images.
# #
# #
# ##Save the trained model as pickle string to disk for future use
# model_name = "sandstone_model_multi_image"
# pickle.dump(model, open(model_name, 'wb'))
# #
# ##To test the model on future datasets
# # loaded_model = pickle.load(open(model_name, 'rb'))
#
#
#
