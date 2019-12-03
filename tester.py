import os
import math
import sys
import time
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np
import lime
from skimage.segmentation import mark_boundaries
from lime import lime_image
from keras.models import load_model
class LimeTester():
 def __init__(self):
  self.testImagePath="./dataset/test"
  self.modelPath="./handrecognition_model.h5"
  self.model=load_model(self.modelPath)
  #xplot and yplot are used for saving the x-axis and y-axis values for the model preditions
  self.xplot=[] 
  self.yplot=[]
  #timex and timey are used for saving the x-axis and y-axis values for the Lime explanations time
  self.timex=[]
  self.timey=[]
 def plotExplainarAndInferenceTime(self):
     predictionTime=0
     explainerTime=0
     count=0
     explainer = lime_image.LimeImageExplainer()
     pX=[]
     pY=[]
     eX=[]
     eY=[]
     
     
     files=[]
     for path in os.listdir(self.testImagePath):
      files.append(path)
      files.sort()
     for path in files:
         count=count+1
         print("Path:"+path)
         im=cv2.imread(self.testImagePath+"/"+path)
         
         im = cv2.resize(im, (320, 120))
         imo=im
         im=np.expand_dims(im,axis=0)
         t1=time.time()
         predict=self.model.predict(im)
         outLabel=(predict.argmax(axis=-1))[0]
         pX.append(outLabel)
         eX.append(outLabel)
         print("Output:"+str(outLabel))
         
         ptime=time.time()-t1
         pY.append(ptime)
         print("Prediction Time:"+str(ptime))
         predictionTime=predictionTime+ptime
         

         #now calculate the time for lime explainer
         t1=time.time()
         explanation = explainer.explain_instance(imo, self.model.predict, top_labels=5, hide_color=0, num_samples=1000)
         temp, mask = explanation.get_image_and_mask(outLabel,positive_only=True, num_features=3, hide_rest=False)
         etime=time.time()-t1
         eY.append(etime)
         print("Explainer Time:"+str(etime))
         explainerTime=explainerTime+etime
        #  if (count==1):
        #   break
         


         
     fig, ax1=plt.subplots()
     
     print ("pX:"+str(pX)+", pY:"+str(pY))
     print ("eX:"+str(eX)+", pY:"+str(eY))
     ax1.set_ylabel("Model inference Time (s)")
     ax1.set_xlabel("Class Labels")    
     ax1.plot(pX,pY,color='tab:red', marker="o",label="Model Inference")
     ax1.tick_params(axis='y', labelcolor='tab:red')
     ax1.legend(loc=1,bbox_to_anchor=(1,0.57))

     ax2=ax1.twinx()
     ax2.set_ylabel("LIME Explainer Time (s)")
     ax2.plot(pX,eY,color='tab:blue', marker="s",label="LIME Explainer  ")
     ax2.tick_params(axis='y', labelcolor='tab:blue')
     ax2.legend(loc=1,bbox_to_anchor=(1,0.50))
     fig.tight_layout()
     

     averagePredictionTime=(predictionTime/count)+1
     averageExplainerTime=(explainerTime/count)+1
     #averagePredictionTime=math.log10(averagePredictionTime)
     #averageExplainerTime=math.log10(averageExplainerTime)
     
     
    #  print("Average Prediction Time:"+str(averagePredictionTime)+" Average Explainer Time:"+str(averageExplainerTime))
    #  plt.bar(["Average Prediction Time", "Average LIME explainer Time"], [averagePredictionTime,averageExplainerTime])
     
    #  plt.ylabel("Time (s)")
     plt.savefig("figs/prediction-explainer-time.png")



 def runLimeExplanations(self):
  count=0
  for testFile in os.listdir(self.testImagePath):
   count=count+1
   findDot=str(testFile)
   trueLabel=findDot.split(".")[0]
   self.xplot.append(trueLabel)
   img = cv2.imread(self.testImagePath+"/"+testFile) # Reads image and returns np.array
   #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
   img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
   origimg=img
   
   #origimg = origimg.reshape(120, 320, 3)
   
   img=image.img_to_array(img)
   img = np.expand_dims(img,axis=0)
   predictions=self.model.predict(img)
   outputLabel=predictions.argmax(axis=-1)[0]
   print("Input:"+testFile+", shape:"+str(origimg.shape)+", Output:"+str(outputLabel))
   self.yplot.append(outputLabel)
   explainer = lime_image.LimeImageExplainer()
   tmp = time.time()
  # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
   self.timex.append(trueLabel)
   explanation = explainer.explain_instance(origimg, self.model.predict, top_labels=5, hide_color=0, num_samples=1000)
   temp, mask = explanation.get_image_and_mask(outputLabel,positive_only=True, num_features=3, hide_rest=False)
   timeTaken=time.time()-tmp
   self.timey.append(timeTaken)
   print ("Time taken for "+str(testFile)+" is:"+str(timeTaken))
   #plt.clf()
   plt.subplot(3,3,count)
   plt.imshow(mark_boundaries(origimg,mask))
 
  #plt.savefig("figs/"+str(trueLabel)+"-"+str(outputLabel)+"-Lime")
 
  plt.savefig("figs/lime-explanations.png")
  plt.clf()
  plt.bar(self.xplot,self.yplot)
  plt.xlabel("True Label")
  plt.ylabel("Predicted Label")
  #plt.show()
  plt.savefig("figs/model-prediction.png")
  plt.clf()
  plt.plot(self.timex,self.timey)
  plt.xlabel("True Label")
  plt.ylabel("Time taken by LIME explainer")
  #plt.show()
  plt.savefig("figs/lime-time.png")