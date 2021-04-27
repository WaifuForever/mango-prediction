import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import numpy as np
import PIL

import sys 
import random

from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from chart import Chart
from model import Model

import tensorflow as tf



from tensorflow.python.client import device_lib




TRAINING_DIR = "./Data/Training"
VALIDATION_DIR = "./Data/Validation"
PREDICTION_DIR = "./Data/Prediction"
RESULT_DIR = "./Data/Result"

MODEL_DIR = "./Models/"



#data.py
width = 512
height = 512
default_epochs = 3
current_model = None
  
       
     
def evaluate_perfomance(model_name):
    
  hits_g = 0
  hits_r = 0
  total_data = 0
  for filename in enumerate(os.listdir(RESULT_DIR + '/Good/')):
    total_data += 1
    if filename[1].startswith('G'):
      hits_g+=1 


  for filename in enumerate(os.listdir(RESULT_DIR + '/Rotten/')): 
    total_data += 1   
    if filename[1].startswith('R'):
      hits_r+=1   
  
  Chart().precision_chart(model_name, [hits_g, hits_r, total_data])





def track_data():
  for filename in enumerate(os.listdir(TRAINING_DIR + '/Good/')):
    if not filename[1].startswith('G - '):
      src = TRAINING_DIR + '/Good/' + filename[1]    
      dst = TRAINING_DIR + '/Good/G - '+ filename[1]
      os.rename(src, dst)

 

  for filename in enumerate(os.listdir(TRAINING_DIR + '/Rotten/')):
    if not filename[1].startswith('R - '):
      src = TRAINING_DIR + '/Rotten/' + filename[1]
      dst =TRAINING_DIR + '/Rotten/R - '+ filename[1]
      os.rename(src, dst)
   
  
   
  print("the data has been tracked")


#still not working
def use_GPU():
  print(device_lib.list_local_devices())

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():  
  
  model = Model()
  model_name = ""
  while True:

    print("\nTENSORFLOW IMAGE CLASSIFIER")
    print("Select a Option:")
    print("1 - Create a new model")
    print("2 - Select a existing model")
    print("3 - End Program")
    

    option = int(input())

    if option == 1:

      model_name = input("Enter model name: ")

      if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is True:
        while True:
          print("This name is already in use")
          model_name = input("Enter model name: ")
          if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is False or model_name == 'exit':
              break
        
      if model_name != 'exit':

        current_model = model.create_model(5, 1)
      

    elif option == 2:      
      #load_model
      model_name = input("Enter model name: ")
      while True:
        if model_name == 'exit':
          break
        
        if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is False:
          print("File not found!!\n")
          model_name = input("Enter model name: ")

        else:
          break
      
      if model_name != 'exit':        
        current_model = model.load_model(model_name)        
            

    elif option == 3:
      break;
      

    else:
      print("Invalid option!!\n")

    while model_name != 'exit':

      print("Select a Option:")
      print("1 - Train model")
      print("2 - Predict images")
      print("3 - Show Data")
      print("4 - Main Menu.")

      option = int(input())

      if option == 1:
          while True:
            try:
                epochs = input("Enter the number of generations: ")
                if epochs == 'exit':
                  break
                else:
                  epochs = int(epochs)
                  current_model = model.train_model(current_model, model_name, epochs)
                  break
                
            except ValueError:
                epochs = default_epochs
                print("Number of generations set to %d" % default_epochs)
                current_model = model.train_model(current_model, model_name, epochs)
                break

      elif option == 2:
      
        while True:        
          print('\nWhich Data should be analyzed?')     
          print("Select a Option:")
          print("1 - From Training Folder ")
          print("2 - From Prediction Folder")
          op = int(input())
          if op == 1:                
              DIR = (TRAINING_DIR, [TRAINING_DIR + '/Good', TRAINING_DIR + '/Rotten'])
              track_data()
              break;
          elif op == 2:
              DIR = (RESULT_DIR, [RESULT_DIR + '/Good', RESULT_DIR + '/Rotten'])
              break;
          else:
              print("Invalid option!!\n")

        model.predict(current_model, model_name, DIR) 
        
        evaluate_perfomance(model_name)

      elif option == 3:
          while True:        
            print('\nChoose the Data to be shown')     
            print("Select a Option:")
            print("1 - Summary")
            print("2 - Guessing_1")
            print("3 - Guessing_2")
            print("4 - Training_1")
            print("5 - Training_2")
            print("6 - Precision")
            print("7 - Return to previous menu")
            op = int(input())
          
            if op == 1:
                current_model.summary() 
            
            elif op == 7:
                print("\n")
                break;
            else:
                Chart().display_charts(model_name, op)
                    
      elif option == 4:
        current_model = None
        break
            
      else:
        print("Invalid option!!\n")




print(tf.__version__)
#use_GPU()
main()