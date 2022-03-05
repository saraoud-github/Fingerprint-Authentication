import os



#Get the directory of the dataset
directory = r''

categories = ['A', 'L', 'R', 'T', 'W']
data = []

#Loop through all the images of old and young people in Databases Folder
for category in categories:
    path = os.path.join(directory, category)
    label = categories.index(category)