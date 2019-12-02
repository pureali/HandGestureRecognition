import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 

class Trainer():
 def __init__(self):
  print("Training class")
  self.datasetPath="./dataset/leapgestrecog"
 def train_model(self):
  print("Starting the training process, please be patient...")
  cwd=os.getcwd()
  imagepaths=[]
  for root, dirs, files in os.walk(self.datasetPath, topdown=False): 
   for name in files:
     path = os.path.join(root, name)
     print(path)
     if path.endswith("png"): # We want only the images
       imagepaths.append(path)

  
  X = [] 
  Y = [] 

  
  for path in imagepaths:
   img = cv2.imread(path) 
   img = cv2.resize(img, (320, 120)) 
   X.append(img)
   # Processing label in image path
   if (path.find("/")==-1):
       continue
   category=path.split("/")[6]
   label = int(category.split("_")[0][1])
   Y.append(label)

  X = np.array(X, dtype="uint8")
  X = X.reshape(len(imagepaths), 120, 320, 3) 
  Y = np.array(Y)
  print("Images loaded: ", len(X))
  print("Labels loaded: ", len(Y))
  print(Y[0], imagepaths[0])
  ts = 0.3 
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ts, random_state=42)
  model = Sequential()
  model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 3))) 
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu')) 
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
  model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=2, validation_data=(X_test, y_test))
  model.save('handrecognition_model.h5')
  test_loss, test_acc = model.evaluate(X_test, y_test)
  print('Test accuracy: {:2.2f}%'.format(test_acc*100))

  predictions = model.predict(X_test) 
  np.argmax(predictions[0]), y_test[0] 

