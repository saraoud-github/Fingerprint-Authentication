import cv2
import os
from tensorflow import keras
import numpy as np


cwd = os.getcwd()
data_path = os.listdir(cwd)
def get_image_class(img):
    #img_path = r"E:\Uni\FYP2\Fingerprint Authentication\Enrolled Fingerprints\download.jfif"
    model_path = os.path.join(cwd, 'ml-model\content\ml-model')
    SIZE = 256
    categories = ['A', 'L', 'R', 'T', 'W']

    # input_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_fingerprint = cv2.resize(img, (SIZE, SIZE))
    input_data = []
    input_data.append(input_fingerprint)
    input_fingerprint = np.asarray(input_data)

    input_fingerprint= input_fingerprint.astype('float32')
    input_fingerprint /= 255
    # img_label = input("Enter image label: ")
    model = keras.models.load_model(model_path)
    label = model.predict(input_fingerprint)
    #print(img_label)
    img_class = np.argmax(label, axis = 1)

    img_label = categories[img_class[0]]
    return img_label

def get_image_id(img, img_label):
    data_dir = os.path.join(data_path[3], 'Enrolled Fingerprints', img_label)

    input_fingerprint = img
    best_score = 0
    filename = None
    image = None
    kp1, kp2, mp = None, None, None
    flag = False

    for file in [file for file in os.listdir(data_dir)]:
        database_fingerprint = cv2.imread(os.path.join(data_dir, file))
        sift = cv2.SIFT_create()
        if input_fingerprint is not None:
            keypoints_1, descriptors_1 = sift.detectAndCompute(input_fingerprint, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(database_fingerprint, None)

            matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
                                            dict()).knnMatch(descriptors_1, descriptors_2, k=2)
            match_points = []

            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints = 0
            if len(keypoints_1) <= len(keypoints_2):
              keypoints = len(keypoints_1)
            else:
              keypoints = len(keypoints_2)

            if (len(match_points) / keypoints) > best_score:
                best_score = len(match_points)/ keypoints * 100
                filename = file
                image = input_fingerprint
                kp1, kp2, mp = keypoints_1, keypoints_2, match_points
                flag = True


    if flag == True:
        print("% match: ", str(best_score))
        print("Fingerprint ID: " + str(filename))
        return str(filename)

    if flag == False:
        print("Cannot identify fingerprint!")
