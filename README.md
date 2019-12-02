# Explainable-HandGestureRecognition

Explainable-HandGestureRecognition with LIME using Keras in Python

This project uses LIME to interpret a Hand Gesture Recognition Model. This project is part of the coursework by the University of Wurzburg, Germany and the EPSS19 Summer School. The project consists of two phases

1. Training of the CNN based deep learning model and saving the model as h5 extension file.
2. Using the LIME API to interpret the inference of the model for a set of images.

Results:
We plot the following results using Matplotlib.
1. The prediction of the model for a set of images in the test folder of the dataset. (Bar Graph)
2. The time taken by the LIME explainer to explain features for each of the images in the test folder (Vector Graph)

The project contains the following tree hierarchy.

HandGestureRecognition (root folder)
1. ----dataset (folder for placing the dataset)
2. ---------test (folder inside dataset consisting of the test images)
3. ----figs (used for saving graphs)
4. -handrecognition_model.h5 (saved hand gesture recognition model)
5. -main.py (main.py for training the model and testing the LIME explanations)
6. -tester.py (Testor class which uses LIME for explanations)
7. -trainer.py (Trainer class for training the model)


Dataset
The dataset used for trainig is provide by the Leap Motion on the Kaggle URL. This project directory only contains the empty folder dataset, please download the data from the kaggle website and extract it into the dataset folder.
Dataset URL: https://www.kaggle.com/gti-upm/leapgestrecog/data

Usage:
1. Download the folder or Git clone. 
2. [optional] Download the dataset if you want to re-train by following the instructions above and extract it.

The main.py contains three methods given below

1. testLimeExplanations() 
2. trainDataset() 
3. main() #this method is executed first
