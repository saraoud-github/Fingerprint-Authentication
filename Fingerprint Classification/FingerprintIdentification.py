import pyrebase
import firebase_admin
from firebase import firebase
from firebase_admin import credentials, storage
import numpy as np
import cv2
import FingerprintMatching as fp_match

config = {
    "apiKey": "AIzaSyB9NSU9nEYWjyJuocmncTzkEb5QAW0-izg",
    "authDomain": "automated-voting-system.firebaseapp.com",
    "databaseURL": "https://automated-voting-system-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "automated-voting-system",
    "storageBucket": "automated-voting-system.appspot.com",
    "messagingSenderId": "71401419044",
    "appId": "1:71401419044:web:684f638c28acb6d64003b1",
    "measurementId": "G-M115EPTM88"
};

path = config["databaseURL"]
fbcon = firebase.FirebaseApplication(path, None)

firebase = pyrebase.initialize_app(config)
db = firebase.database()

cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, config)
bucket = storage.bucket()

template = db.child("data").get().val()
print(template["finger_id"])
blob = bucket.get_blob(str(template["finger_id"])+ '.png')
arr = np.frombuffer(blob.download_as_string(), np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Image', img)
# cv2.waitKey(0)

# img_path = r"E:\Uni\FYP2\Fingerprint Authentication\Enrolled Fingerprints\1.png"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_label = fp_match.get_image_class(img)
img_id = fp_match.get_image_id(img, img_label)
print("Fingerprint ID:", img_id[0:12])
fbcon.put('data', 'voter_id', int(img_id[0:12]))




