from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
from data import Data
from chart import Chart
import numpy as np
import json
import pathlib
import os
import tensorflow as tf
import h5py
import msvcrt

class Model:

    PREDICTION_DIR = "./Data/Prediction"
    RESULT_DIR = "./Data/Result"
    TRAINING_DIR = "./Data/Training"
    VALIDATION_DIR = "./Data/Validation"
    MODEL_DIR = "./Models/"
    height, width = 512, 512

    #model.py
    batch_size = 4
    default_epochs = 3

    def __init__(self):
        pass

    def _get_RNN(self, op):
        if op == 1:
            return Sequential([     
                
                layers.Conv2D(16, (5, 5), activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(32, (5, 5), activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(64, (5, 5), activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(128, (5, 5), activation="relu"),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(256, (5, 5), activation="relu"),
                layers.MaxPooling2D(2,2),
                layers.Dropout(0.2),
                
                layers.Conv2D(512, (5, 5), activation="relu"),      
                layers.MaxPooling2D(2,2),

                layers.Flatten(),   
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                
                layers.Dense(1, activation='sigmoid'),
                
            ]), 
        elif op == 2:
            #VGG16
            model = Sequential()
            model.add(layers.ZeroPadding2D((1,1),input_shape=(self.height, self.width, 3)))
            model.add(layers.Convolution2D(64, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(64, 3, 3, activation='relu'))
            model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(128, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(128, 3, 3, activation='relu'))
            model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
            model.add(layers.MaxPooling2D((2,2), padding='same', strides=(2,2)))

            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.MaxPooling2D((2,2), padding='same', strides=(2,2)))

            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
            model.add(layers.MaxPooling2D((2,2), padding='same', strides=(2,2)))

            model.add(layers.Flatten())
            model.add(layers.Dense(4096, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(4096, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(2, activation='softmax'))
            return model

    def create_model(self):
    
        model = self._get_RNN(2)

        opt = tf.keras.optimizers.RMSprop()

        #sparse_categorical_crossentropy
        #binary_crossentropy
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        model.summary()
           
        return model
        #model = train_data(None, model_name, epochs)     


        #model.save(MODEL_DIR + model_name + '/'+ model_name)
        #Chart().training_chart(model_name)
        
        #predict(model, model_name)
        #evaluate_perfomance(model_name)
        

    def train_model(self, model, model_name, epochs):  
        
        DIR = pathlib.Path(self.TRAINING_DIR)
        #Data().resize_images(self.width, self.height)
        
        gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
            height_shift_range=0.1, zoom_range=0.1, rescale=1./255,
            channel_shift_range=10, horizontal_flip=True, fill_mode="nearest")

        test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.7)

        train_generator = gen.flow_from_directory(
            self.TRAINING_DIR,
            target_size=(self.width, self.height),
            batch_size=self.batch_size, 
            shuffle=True,       
            subset='training',        
            class_mode='binary',
            save_format='.jpeg',
            save_to_dir="./Data/Augmented",
            save_prefix=model_name + " - "
        )

        validation_generator = test_datagen.flow_from_directory(
            self.VALIDATION_DIR,
            target_size=(self.width, self.height),
            shuffle=True, 
            batch_size=self.batch_size//2,            
            subset='validation',
            class_mode='binary'
        )

        print(train_generator.image_shape)
        print(validation_generator.image_shape)

        print(train_generator)
        print(validation_generator)

        class_names = []
        for key in train_generator.class_indices.keys():    
            class_names.append(key)

        print(class_names)
        
        history = model.fit(
            train_generator, 
            steps_per_epoch=419 // self.batch_size,       
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=281 // (self.batch_size // 2)
        )

        
        data = ({
            'acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'epochs_range' : epochs
        })

        self._save_model(model_name, model, data)
     

        return model

    def _save_model(self, model_name, model, data):
        try:
            os.mkdir(self.MODEL_DIR + model_name + '/')
        except OSError:
            print ("Creation of the directory for %s failed" % model_name)
        else:
            print ("Successfully created the directory for %s " % model_name)
        
        model_json = model.to_json()
        with open(self.MODEL_DIR + model_name + '/'+ model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.MODEL_DIR + model_name + '/'+ model_name + ".h5")
        print("Saved model to disk\n")

        if os.path.isfile(self.MODEL_DIR + model_name + '/'+ model_name + '.h5') is False:
            model.save_weights(self.MODEL_DIR + model_name + '/'+ model_name + ".h5")
        
        
        with open(self.MODEL_DIR + model_name + '/' + model_name + '_data.txt', 'w') as outfile:
            json.dump(data, outfile)


    def predict(self, model, model_name, DIR):    

        
        while True:        
            print('\nDo you want to erase the current Data in Result folder?')     
            print("Select a Option:")
            print("1 - Yes ")
            print("2 - No")
            op = int(input())
            if op == 1:
            
                for path in [self.RESULT_DIR + '/Good', self.RESULT_DIR + '/Rotten']:

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

        
        class_names = [x[1]for x in os.walk(self.RESULT_DIR)][0]
        print(class_names )

        scores = {
        "Good": [],
        "Rotten": [],
        "Average": []
        }
        
        for path in DIR:

            onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            image_count = len(list(onlyfiles))

            for x in range (0, image_count):
            

                current_path = path + '/' + onlyfiles[x]
                
                img = keras.preprocessing.image.load_img(
                    current_path, target_size=(self.height, self.width)
                )
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

               
               
           
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                print(predictions)
                print(score)

                if(np.argmax(score) == 0):           
                    copyfile(current_path, self.RESULT_DIR + '/Good/' + onlyfiles[x])
                    scores["Good"].append(np.max(score))
                else:
                    copyfile(current_path, self.RESULT_DIR + '/Rotten/' + onlyfiles[x])
                    scores["Rotten"].append(np.max(score))

                scores["Average"].append(np.max(score))
                
                print(
                    "This image ({}) most likely belongs to {} with a {:.2f} percent confidence."
                    .format(onlyfiles[x], class_names[np.argmax(score)], 100 * np.max(score))
                )
                
        Chart().guessing_chart(model_name, scores)

    def load_model(self, model_name):
        if os.path.isfile(self.MODEL_DIR + model_name + '/' + model_name +  '.h5') is True:
            json_file = open(self.MODEL_DIR + model_name + '/' + model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(self.MODEL_DIR + model_name + '/' + model_name + '.h5')
            
            print("Loaded model from disk")

            return loaded_model
        
        elif model_name == 'exit':
            print("Back to main menu")
            pass

        else:    
            while True:
                print("Model not found!")
                model_name = input("Enter model name: ")
                if model_name == 'exit':
                    break;
                elif os.path.isfile(self.MODEL_DIR + model_name + '/' + model_name + '.h5') is True:
                    json_file = open(self.MODEL_DIR + model_name + '/' + model_name + '.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    loaded_model = model_from_json(loaded_model_json)
                    # load weights into new model
                    loaded_model.load_weights(self.MODEL_DIR + model_name + '/' + model_name + '.h5')
                    print("Loaded model from disk\n")

                    return loaded_model
                    
        
       
