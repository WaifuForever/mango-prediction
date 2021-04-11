import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import json
import sys 

from chart import Chart

import tensorflow as tf
import h5py
import msvcrt
from shutil import copyfile

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


from tensorflow.python.client import device_lib




TRAINING_DIR = "./Data/Training"
PREDICTION_DIR = "./Data/Prediction"
RESULT_DIR = "./Data/Result"

MODEL_DIR = "./Models/"



default_epochs = 3
width = 686
height = 656
batch_size = 32
total_training_data = 0



def load_model(model_name):
  if os.path.isfile(MODEL_DIR + model_name + '/' + model_name +  '.h5') is True:
    json_file = open(MODEL_DIR + model_name + '/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(MODEL_DIR + model_name + '/' + model_name + '.h5')
    
    print("Loaded model from disk")

  else:    
    while True:
      print("Model not found!")
      model_name = input("Enter model name: ")
      if model_name == 'exit':
        break;
      if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is True:
        json_file = open(MODEL_DIR + model_name + '/' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODEL_DIR + model_name + '/' + model_name + '.h5')
        print("Loaded model from disk")

        
        loaded_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        break;
  
  if model_name != 'exit':
    while True:
      
      print("Select a Option:")
      print("1 - Retrain model")
      print("2 - Predict images")
      print("3 - Show Data")
      print("4 - Main Menu.")

      option = int(input())

      if option == 1:
        while True:
          try:
            epochs = int("Enter the number of generations: ")
            break
            
          except ValueError:
            epochs = default_epochs
            print("Number of generations set to %d" % default_epochs)
            break
  
        loaded_model = train_data(loaded_model, model_name, epochs)

      elif option == 2:
        predict(loaded_model)
        evaluate_perfomance(model_name)
      elif option == 3:
        loaded_model.summary()        
       
        
        # open method used to open different extension image file
        try:
          im = PIL.Image.open(MODEL_DIR + model_name + '/'+ model_name + '.png')        
          im.show()
        except:
          Chart().train_chart(model_name)
        
        try:       
          im = PIL.Image.open(MODEL_DIR + model_name + '/'+ model_name + '_result' + '.png') 
          im.show() 
        except:
          print("The file %s could not be found\n" % (model_name + '_result' + '.png'))
      
      elif option == 4:
        
        break
          
      else:
        print("Invalid option!!\n")
    

    if os.path.isfile(MODEL_DIR + model_name + '/'+ model_name + '.h5') is False:
      loaded_model.save_weights(MODEL_DIR + model_name + '/'+ model_name + ".h5")
      

    if os.path.isfile(MODEL_DIR + model_name + '/'+ model_name + '.json') is False:
      model_json = loaded_model.to_json()
      with open(MODEL_DIR + model_name + '/'+ model_name + ".json", "w") as json_file:
        json_file.write(model_json)
          

def create_model(model_name):
  if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is True or model_name == 'exit':
    while True:
      print("This name is already in use")
      model_name = input("Enter model name: ")
      if os.path.isfile(MODEL_DIR + model_name + '/' + model_name + '.h5') is False or model_name == 'exit':
        break
  
  if model_name != 'exit':    
    
    while True:
      try:
        epochs = int("Enter the number of generations: ")
        break
        
      except ValueError:
        epochs = default_epochs
        print("Number of generations set to %d" % default_epochs)
        break
    
   
    model = train_data(None, model_name, epochs)
  
     

    #model.save(MODEL_DIR + model_name + '/'+ model_name)
    Chart().train_chart(model_name)
    
    predict(model)
    #evaluate_perfomance(model_name)
  
  
def resize_images():
  total_training_data = 0
  good_images = []
  rotten_images = []
  for f in os.listdir(TRAINING_DIR+'/Good'):
    if os.path.isfile(os.path.join(TRAINING_DIR+'/Good', f)):   
      image = PIL.Image.open(TRAINING_DIR+'/Good/'+ f)
      new_image = image.resize((width, height))
      new_image.save(TRAINING_DIR+'/Good/' + f)
      good_images.append(f)
      total_training_data+=1
    

  for f in os.listdir(TRAINING_DIR+'/Rotten'):
    if os.path.isfile(os.path.join(TRAINING_DIR+'/Rotten', f)):   
      image = PIL.Image.open(TRAINING_DIR+'/Rotten/' + f)
      new_image = image.resize((width, height))
      new_image.save(TRAINING_DIR+'/Rotten/' + f)   
      rotten_images.append(f)
      total_training_data+=1
    

def train_data(model, model_name, epochs):  
  
  global class_names

  DIR = pathlib.Path(TRAINING_DIR)
  resize_images()
  
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)


  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)

  class_names = train_ds.class_names
  print(class_names)

  
  data_augmentation = keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                  input_shape=(height, 
                                                                width,
                                                                3)),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
  )
  if model == None:
    num_classes = 2
    #THIS CURRENT MODEL IS NOT WORKING!!
    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(height, width, 3)),
      data_augmentation,
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])



  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


  model.summary()
  
  history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
  
  )


  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  try:
    os.mkdir(MODEL_DIR + model_name + '/')
  except OSError:
    print ("Creation of the directory for %s failed" % model_name)
  else:
    print ("Successfully created the directory for %s " % model_name)
  
  model_json = model.to_json()
  with open(MODEL_DIR + model_name + '/'+ model_name + ".json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(MODEL_DIR + model_name + '/'+ model_name + ".h5")
  print("Saved model to disk\n")

  if os.path.isfile(MODEL_DIR + model_name + '/'+ model_name + '.h5') is False:
    model.save_weights(MODEL_DIR + model_name + '/'+ model_name + ".h5")
  
  
  data = ({
      'acc': acc,
      'val_acc': val_acc,
      'loss': loss,
      'val_loss': val_loss,
      'epochs_range' : default_epochs
  })

  with open(MODEL_DIR + model_name + '/' + model_name + '_data.txt', 'w') as outfile:
      json.dump(data, outfile)

  return model

def predict(model): 
  
  while True:        
      print('\nWhich Data should be analyzed?')     
      print("Select a Option:")
      print("1 - From Training Folder ")
      print("2 - From Prediction Folder")
      op = int(input())
      if op == 1:
        
        DIR = [TRAINING_DIR + '/Good', TRAINING_DIR + '/Rotten']
        break;
      elif op == 2:
        DIR = [PREDICTION_DIR]
        break;
      else:
        print("Invalid option!!\n")
  
  while True:        
      print('\nDo you want to erase the current Data in Result folder?')     
      print("Select a Option:")
      print("1 - Yes ")
      print("2 - No")
      op = int(input())
      if op == 1:
       
        for path in [RESULT_DIR + '/Good', RESULT_DIR + '/Rotten']:

          onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
          image_count = len(list(onlyfiles))

          for x in range (0, image_count):
            
            current_path = path + '/' + onlyfiles[x]
            
            
            try:
              if os.path.isfile(current_path) or os.path.islink(current_path):
                  os.unlink(current_path)
              elif os.path.isdir(current_path):
                  shutil.rmtree(current_path)
            except Exception as e:
              print('Failed to delete %s. Reason: %s' % (current_path, e))

        print('\nThe Current Data in Result folder has been deleted')  
     
  
        break;
      elif op == 2:
        print('\nThe Current Data in Result folder has not been deleted')
        break;
      else:
        print("Invalid option!!\n")

  
  class_names = [x[1]for x in os.walk(RESULT_DIR)][0]
  print(class_names)



  for path in DIR:

    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    image_count = len(list(onlyfiles))

    for x in range (0, image_count):
      
      current_path = path + '/' + onlyfiles[x]
     
      img = keras.preprocessing.image.load_img(
          current_path, target_size=(height, width)
      )
      img_array = keras.preprocessing.image.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0) # Create a batch

      predictions = model.predict(img_array)
      score = tf.nn.softmax(predictions[0])

      if(np.argmax(score) == 0):             

        copyfile(current_path, RESULT_DIR + '/Good/' + onlyfiles[x])
        
      else:
        copyfile(current_path, RESULT_DIR + '/Rotten/' + onlyfiles[x])
      
     
      print(
          "This image ({}) most likely belongs to {} with a {:.2f} percent confidence."
          .format(onlyfiles[x], class_names[np.argmax(score)], 100 * np.max(score))
      )
  

def evaluate_perfomance(model_name):
    
  hits_g = 0
  hits_r = 0

  for filename in enumerate(os.listdir(RESULT_DIR + '/Good/')):
    if filename[1].startswith('G'):
      hits_g+=1 


  for filename in enumerate(os.listdir(RESULT_DIR + '/Rotten/')):    
    if filename[1].startswith('R'):
      hits_r+=1 
  

  p1 = Chart()
  
  p1.hits_chart(model_name, [hits_g, hits_r, total_training_data])



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

  #physical_devices = tf.config.experimental.list_physical_devices('GPU')
  print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
  #tf.config.experimental.set_memory_growth(physical_devices[0], True)


print(tf.__version__)
#use_GPU()

while True:

  print("\nTENSORFLOW IMAGE CLASSIFIER")
  print("Select a Option:")
  print("1 - Create a new model")
  print("2 - Select a existing model")
  print("3 - Track training data")
  print("4 - End Program")

  option = int(input())

  if option == 1:
    create_model(input("Enter model name: "))
  elif option == 2:
    load_model(input("Enter model name: "))
  elif option == 3:
    track_data()
  elif option == 4:
    break;
  
  else:
    print("Invalid option!!\n")

