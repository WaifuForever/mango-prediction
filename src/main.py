import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import numpy as np
import PIL

import sys 
import random
import csv

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
batch_size = 2
current_model = None
  
       
def track_data():
  with open(TRAINING_DIR + '/validation_result.csv', mode='w') as data_file:
    data_writer = csv.writer(data_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    headerList = ['filename', 'output']
    data_writer.writerow(headerList)
    for filename in enumerate(os.listdir(TRAINING_DIR + '/Good/')):
      data_writer.writerow([filename, 0])

  
    for filename in enumerate(os.listdir(TRAINING_DIR + '/Rotten/')):
      data_writer.writerow([filename, 1])
   
  
   
  print("the data has been tracked")

def run_rotine():
  model = Model()

  names = [
    "LeNet-5",
    "AlexNet",
    "VGG16",
    "relu-sigmoid[3-512]",
    "relu-linear-sigmoid[2-128]",
    "relu-sigmoid[4-2048]",
    "sigmoid+BN[2+1/2-256]",
    "sigmoid+BN[4-4096]",
    "sigmoid+BN[2+1/2-512]"
  ]

  optimizers = [
    " - RMSprop",
    " - Adam",
    " - SGD",
    " - Adadelta",
    " - Adagrad",
    " - Adamax",
    " - Nadam",
    " - Ftrl",    
  ]

  batch_size = [2, 4, 8, 16]
  pool_size = [(1, 1), (3, 3), (5, 5), (7, 7)]
  learning_rate = [0.01, 0.001, 0.0001, 0.00001]
  DIR = (TRAINING_DIR, [TRAINING_DIR + '/Good', TRAINING_DIR + '/Rotten'])
  epochs = 2
  for k in batch_size:
      for x in pool_size:
          for y in range(0, 9):
              for z in learning_rate:
                for opt in range (0, 8):
                  m1 = model.create_model(y, x, z, opt)
                  m1 = model.train_model(m1, names[y], epochs, k)
                  model.predict(m1, names[y], DIR, False)
                  with open(self.MODEL_DIR + names[y] + '/info.csv', mode='w') as data_file:
                    headerList = ['model_name', 'batch_size', 'pool_size', 'learning_rate']
                    data_writer = csv.writer(data_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(headerList)
                    info = [names[y] + optimizers[opt], k, x, z]
                    data_writer.writerow(info)



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
        cnn = 7
        op = 2
        lr = 0.001
        pool_size = (3,3)
        current_model = model.create_model(cnn, pool_size, lr, op)
      

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
                  current_model = model.train_model(current_model, model_name, epochs, batch_size)
                  break
                
            except ValueError:
                epochs = default_epochs
                print("Number of generations set to %d" % default_epochs)
                current_model = model.train_model(current_model, model_name, epochs, batch_size)
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

        model.predict(current_model, model_name, DIR, True) 
        
        

      elif option == 3:
          while True:        
            print('\nChoose the Data to be shown')     
            print("Select a Option:")
            print("1 - Summary")
            print("2 - Architecture")           
            print("3 - Training_1")
            print("4 - Training_2")
            print("5 - Assurance_1")
            print("6 - Assurance_2")
            print("7 - Assurance_3")                       
            print("8 - Precision")
            print("9 - Return to previous menu")
            op = int(input())

            switcher = {
              2: "_model_plot",
              3: "_training_1",
              4: "_training_2",
              5: "_assurance_1",
              6: "_assurance_2",
              7: "_assurance_3",
              8: "_precision",
                            
            }
          
            if op == 1:
                current_model.summary() 
            
            elif op == 9:
                print("\n")
                break;
            else:
                Chart().display_chart(model_name, switcher[op], True)
                    
      elif option == 4:
        current_model = None
        break
            
      else:
        print("Invalid option!!\n")




print(tf.__version__)
#use_GPU()
run_rotine()
#main()