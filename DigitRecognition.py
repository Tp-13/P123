import cv2
import numpy as np
import pandas as pd
from pandas.core import frame
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
#print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

#Splitting the data and scaling it
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
x_train_scale = x_train/255
x_test_scale = x_test/255

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scale, y_train)
y_pred = clf.predict(x_test_scale)

#Calculating the accuracy of the modal
Accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", Accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))
        print(upper_left)
        print(bottom_right)
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        #roi = Region of Interest
        #to consider the area inside the box for detecting the digit
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)
        img_bw = im_pil.convert('L')
        img_bw_resized = img_bw.resize((22, 30))
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
        test_pred = clf.predict(test_sample)
        print("Predicted Letter is ", test_pred)
        
        #displaying the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()