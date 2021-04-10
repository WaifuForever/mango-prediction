import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL


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



default_epochs = 15
width = 686
height = 656
batch_size = 32




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
      print("4 - End Program")

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
  
        loaded_model = train_data(loaded_model, epochs)

      elif option == 2:
        predict(loaded_model)
      elif option == 3:
        loaded_model.summary()        
  
        # open method used to open different extension image file
        im = PIL.Image.open(MODEL_DIR + model_name + '/'+ model_name + '.png') 
          
        # This method will show image in any image viewer 
        im.show() 
       
        
      
      elif option == 4:
        print("End Program.")
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
  if os.path.isfile(MODEL_DIR + model_name + '.h5') is True or model_name == 'exit':
    while True:
      print("This name is already in use")
      model_name = input("Enter model name: ")
      if os.path.isfile(MODEL_DIR + model_name + '.h5') is False or model_name == 'exit':
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
  
    model = train_data(None, epochs)
  
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
    print("Saved model to disk")

    if os.path.isfile(MODEL_DIR + model_name + '/'+ model_name + '.h5') is False:
      model.save_weights(MODEL_DIR + model_name + '/'+ model_name + ".h5")
     

    #model.save(MODEL_DIR + model_name + '/'+ model_name)

    show_data(model.history, epochs, model_name)
    predict(model)
  
  
def resize_images():
  good_images = []
  rotten_images = []
  for f in os.listdir(TRAINING_DIR+'/Good'):
   if os.path.isfile(os.path.join(TRAINING_DIR+'/Good', f)):   
    image = PIL.Image.open(TRAINING_DIR+'/Good/'+ f)
    new_image = image.resize((width, height))
    new_image.save(TRAINING_DIR+'/Good/' + f)
    good_images.append(f)

  for f in os.listdir(TRAINING_DIR+'/Rotten'):
   if os.path.isfile(os.path.join(TRAINING_DIR+'/Rotten', f)):   
    image = PIL.Image.open(TRAINING_DIR+'/Rotten/' + f)
    new_image = image.resize((width, height))
    new_image.save(TRAINING_DIR+'/Rotten/' + f)   
    rotten_images.append(f)
  
  print()

def train_data(model, epochs):  
  
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
  return model

def show_data(history, epochs, model_name):

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.savefig(MODEL_DIR + model_name + '/'+ model_name +'.png')
 
  plt.show()

def predict(model):
  
  DIR = PREDICTION_DIR
  onlyfiles = [f for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))]
  image_count = len(list(onlyfiles))

  for x in range (0, image_count):

    
    current_path = DIR + '/' + onlyfiles[x]
    image = PIL.Image.open(current_path)
    
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
    
    print(class_names)
    print(np.argmax(score))
    print(
        "This image ({}) most likely belongs to {} with a {:.2f} percent confidence."
        .format(onlyfiles[x], class_names[np.argmax(score)], 100 * np.max(score))
    )


#still not working
def use_GPU():
  print(device_lib.list_local_devices())

  #physical_devices = tf.config.experimental.list_physical_devices('GPU')
  print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
  #tf.config.experimental.set_memory_growth(physical_devices[0], True)


print(tf.__version__)
#use_GPU()

while True:

  print("TENSORFLOW IMAGE CLASSIFIER")
  print("Select a Option:")
  print("1 - Create a new model")
  print("2 - Select a existing model")
  print("3 - End Program")

  option = int(input())

  if option == 1:
    create_model(input("Enter model name: "))
  elif option == 2:
    load_model(input("Enter model name: "))
  elif option == 3:
    break;
  
  else:
    print("Invalid option!!\n")

