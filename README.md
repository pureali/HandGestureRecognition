# Explainable-HandGestureRecognition

Explainable-HandGestureRecognition with LIME using Keras in Python

This project uses LIME to interpret a Hand Gesture Recognition Model. This project is motivated by the european conference on machine learning summer school (EPSS 19 Wurzburg, Germany 11th-16 Sep 19) and is part of the coursework by the University of Wurzburg, Germany. The project consists of two phases

1. Training of the CNN based deep learning model and saving the model as h5 extension file.
2. Using the LIME API to interpret the inference of the model for a set of images.

Results:
We plot the following results using Matplotlib.
1. The prediction of the model for a set of images in the test folder of the dataset. (Bar Graph)
2. The time taken by the LIME explainer to explain features for each of the images in the test folder (Vector Graph)

The project contains the following tree hierarchy.

HandGestureRecognition (root folder)
----dataset (folder for placing the dataset)
---------test (folder inside dataset consisting of the test images)
----figs (used for saving graphs)
-handrecognition_model.h5 (saved hand gesture recognition model)
-main.py (main.py for training the model and testing the LIME explanations)
-tester.py (Testor class which uses LIME for explanations)
-trainer.py (Trainer class for training the model)

Dataset
The dataset used for trainig is provide by the Leap Motion on the Kaggle URL. This project directory only contains the empty folder dataset, please download the data from the kaggle website and extract it into the dataset folder.
Dataset URL: https://www.kaggle.com/gti-upm/leapgestrecog/data

Usage:
Download the folder or Git clone. Then download the dataset by following the instructions above and extract it.
The main.py contains three methods given below

testLimeExplanations() 
trainDataset() 
main() #this method is executed first

