from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix
from shutil import copyfile
from chart import Chart

import numpy as np
import pathlib
import os
import tensorflow as tf
import h5py
import msvcrt
import csv

class Model:

    PREDICTION_DIR = "./Data/Prediction"
    RESULT_DIR = "./Data/Result"
    TRAINING_DIR = "./Data/Training"
    VALIDATION_DIR = "./Data/Validation"
    MODEL_DIR = "./Models/"
    height, width = 400, 400

   

    def __init__(self):
        pass


    def _get_optimizer(self, op, learning_rate):                
        # map the inputs to the function blocks
        return {
            0 : tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            1 : tf.keras.optimizers.Adam(learning_rate=learning_rate),
            2 : tf.keras.optimizers.SGD(learning_rate=learning_rate),
            3 : tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
            4 : tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
            5 : tf.keras.optimizers.Adamax(learning_rate=learning_rate),
            6 : tf.keras.optimizers.Nadam(learning_rate=learning_rate),
            7 : tf.keras.optimizers.Ftrl(learning_rate=learning_rate),
        }.get(op, None) 
        
    def _get_CNN(self, pool_size, op):

        return { 
            #LeNet-5            
            0 : Sequential([     
               
                layers.Conv2D(32, pool_size, activation="sigmoid", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D((2,2), strides=2),

                layers.Conv2D(48, pool_size, activation="sigmoid", padding="valid"),
                layers.MaxPooling2D((2,2), strides=2),
                
                layers.Flatten(),   
                layers.Dense(256, activation="sigmoid"),
                layers.Dropout(0.2),
                layers.Dense(84, activation="sigmoid"),
                layers.Dense(1, activation="sigmoid"),
            
            
                ]),

            #AlexNet
            1 : Sequential([     
                                                                
                #1st Convolutional Layer
                layers.Conv2D(filters=96, activation='relu', input_shape=(self.height, self.width, 3), kernel_size=(11,11), strides=(4,4), padding='same'),
                layers.BatchNormalization(),               
                layers.MaxPooling2D(pool_size=pool_size, strides=(2,2), padding='same'),


                #2nd Convolutional Layer
                layers.Conv2D(filters=256, activation='relu', kernel_size=(5, 5), strides=(1,1), padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=pool_size, strides=(2,2), padding='same'),

                #3rd Convolutional Layer
                layers.Conv2D(filters=384, activation='relu', kernel_size=(3,3), strides=(1,1), padding='same'),
                layers.BatchNormalization(),
                
                #4th Convolutional Layer
                layers.Conv2D(filters=384, activation='relu', kernel_size=(3,3), strides=(1,1), padding='same'),
                layers.BatchNormalization(),
                

                #5th Convolutional Layer
                layers.Conv2D(filters=256, activation='relu', kernel_size=(3,3), strides=(1,1), padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=pool_size, strides=(2,2), padding='same'),

                #Passing it to a Fully Connected layer
                layers.Flatten(),
                # 1st Fully Connected Layer
                layers.Dense(4096, activation='relu',  input_shape=(self.height, self.width, 3)),
                layers.BatchNormalization(),
                # Add Dropout to prevent overfitting
                layers.Dropout(0.4),

                #2nd Fully Connected Layer
                layers.Dense(4096, activation='relu'),
                layers.BatchNormalization(),
                #Add Dropout
                layers.Dropout(0.4),

                #3rd Fully Connected Layer
                layers.Dense(1000, activation='relu'),
                layers.BatchNormalization(),
                #Add Dropout
                layers.Dropout(0.4),

                #Output Layer
                layers.Dense(1, activation='sigmoid'),
                #layers.BatchNormalization())
               
                
                ]),
            #VGG16
            2 : Sequential([
                    layers.ZeroPadding2D(pool_size,input_shape=(self.height, self.width, 3)),
                    layers.Convolution2D(64, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(64, 3, 3, activation='relu'),
                    layers.MaxPooling2D((2,2), strides=(2,2)),

                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(128, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(128, 3, 3, activation='relu'),
                    layers.MaxPooling2D((2,2), strides=(2,2)),

                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(256, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(256, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(256, 3, 3, activation='relu'),
                    layers.MaxPooling2D((2,2), padding='same', strides=(2,2)),

                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.MaxPooling2D((2,2), padding='same', strides=(2,2)),

                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.ZeroPadding2D(pool_size),
                    layers.Convolution2D(512, 3, 3, activation='relu'),
                    layers.MaxPooling2D((2,2), padding='same', strides=(2,2)),

                    layers.Flatten(),
                    layers.Dense(4096, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(4096, activation='linear'),                    
                    layers.Dropout(0.5),
                    layers.Dense(1, activation='sigmoid'),

                ]),

            #relu-sigmoid[3-512]    broken?
            3 : Sequential([     
                
                layers.Conv2D(16, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(32, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),

                layers.Conv2D(64, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(128, pool_size, activation="relu"),
                layers.MaxPooling2D(2,2),

                layers.Conv2D(256, pool_size, activation="relu"),
                layers.MaxPooling2D(2,2),
                layers.Dropout(0.2), 
                layers.Conv2D(512, pool_size, activation="relu"),
                layers.MaxPooling2D(2,2),               

                layers.Flatten(),   
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                
                layers.Dense(1, activation='sigmoid'),
                
                ]),

            #relu-linear-sigmoid[2-128] broken?
            4 : Sequential([     
                
                layers.Conv2D(16, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(32, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),

                layers.Conv2D(64, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(128, pool_size, activation='linear'),                              
                layers.MaxPooling2D(2,2),
                

                layers.Flatten(),   
                layers.Dense(128, activation='linear'),
                layers.Dropout(0.5),                
                layers.Dense(1, activation='sigmoid'),
                
                ]),

            #relu-sigmoid[4-2048]  
            5 : Sequential([
                
                layers.Conv2D(16, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),            
                layers.Conv2D(32, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(), 

                layers.Conv2D(64, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),            
                layers.Conv2D(128, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(), 

                layers.Conv2D(256, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),            
                layers.Conv2D(512, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(), 

                layers.Conv2D(1024, pool_size, activation="relu", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),            
                layers.Conv2D(2048, pool_size, activation="relu", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),           
                

                layers.Flatten(),   
                layers.Dense(2048, activation="relu"),
                layers.Dropout(0.5),                
                layers.Dense(1, activation='sigmoid'),
                
                ]),
            #sigmoid+BN[2+1/2-256] good results
            6 : Sequential([     
                
                layers.Conv2D(16, pool_size, activation="sigmoid", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),
                layers.Conv2D(32, pool_size, activation='sigmoid', padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),

                layers.Conv2D(64, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),
                layers.Conv2D(128, pool_size, activation="sigmoid"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),

                layers.Conv2D(256, pool_size, activation="sigmoid"),
                layers.MaxPooling2D(2,2),
                layers.Dropout(0.2),
                layers.BatchNormalization(),                

                layers.Flatten(),   
                layers.Dense(256, activation="sigmoid"),
                layers.Dropout(0.5),
                
                layers.Dense(1, activation='sigmoid'),
                
                ]),
            #sigmoid+BN[4-4096] broken
            7 : Sequential([     
                layers.Conv2D(32, pool_size, activation="sigmoid", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),               
                layers.Conv2D(64, pool_size, activation='sigmoid', padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),
                
                layers.Conv2D(128, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),             
                layers.Conv2D(256, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),

                layers.Conv2D(512, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),                
                layers.Conv2D(1024, pool_size, activation='sigmoid', padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),            

                layers.Conv2D(2048, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),                
                layers.Conv2D(4096, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),
                layers.BatchNormalization(),                             
                layers.Dropout(0.2),                                

                layers.Flatten(),   
                layers.Dense(4096, activation="sigmoid"),
                layers.Dropout(0.5),
                
                layers.Dense(1, activation='sigmoid'),
               
                
                ]),
            #sigmoid+BN[2+1/2-512]  broken?          
            8 : Sequential([     
                
                layers.Conv2D(32, pool_size, activation="sigmoid", padding="same", input_shape=(self.height, self.width, 3)),
                layers.MaxPooling2D(2,2),                
                layers.Conv2D(64, pool_size, activation='sigmoid', padding="same"),
                layers.MaxPooling2D(2,2),
                
                layers.Conv2D(128, pool_size, activation="sigmoid", padding="same"),
                layers.MaxPooling2D(2,2),                
                layers.Conv2D(256, pool_size, activation="sigmoid"),
                layers.MaxPooling2D(2,2),
                

                layers.Conv2D(512, pool_size, activation="sigmoid"),
                layers.MaxPooling2D(2,2),
                layers.Dropout(0.2),
                layers.BatchNormalization(),                

                layers.Flatten(),   
                layers.Dense(512, activation="sigmoid"),
                layers.Dropout(0.5),
                
                layers.Dense(1, activation='sigmoid'),
                
                ]),
            
        }.get(op, None) 

    def _save_model(self, model_name, model):
        try:
            os.mkdir(self.MODEL_DIR + model_name + '/')
        except OSError:
            print ("Creation of the directory for %s failed" % model_name)
        else:
            print ("Successfully created the directory for %s " % model_name)
            
        keras.utils.plot_model(
            model,
            to_file=self.MODEL_DIR + model_name + '/' + model_name + '_model_plot.png',
            show_shapes=True,
            show_layer_names=True
        )

        model_json = model.to_json()
        with open(self.MODEL_DIR + model_name + '/'+ model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.MODEL_DIR + model_name + '/'+ model_name + ".h5")
        print("Saved model to disk\n")

        if os.path.isfile(self.MODEL_DIR + model_name + '/'+ model_name + '.h5') is False:
            model.save_weights(self.MODEL_DIR + model_name + '/'+ model_name + ".h5")


    def create_model(self, cn, pool_size, lr, opt):
    
        model = self._get_CNN((3,3), cn)

        #sparse_categorical_crossentropy
        #binary_crossentropy
        model.compile(optimizer=self._get_optimizer(opt, lr),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        model.summary()
          
        return model
        
        
    def train_model(self, model, model_name, epochs, batch_size):  
        
        DIR = pathlib.Path(self.TRAINING_DIR)        
        
        gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
            height_shift_range=0.05, rescale=1./255, brightness_range=(0.3, 0.7),
            channel_shift_range=10, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

        test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.7)

        train_generator = gen.flow_from_directory(
            self.TRAINING_DIR,
            target_size=(self.width, self.height),
            batch_size=batch_size, 
            shuffle=True,       
            color_mode="rgb",
            subset='training',        
            class_mode='binary'           
        )

        '''
        save_format='.jpeg',
        save_to_dir="./Data/Augmented",
        save_prefix=model_name + " - "
        '''

        validation_generator = test_datagen.flow_from_directory(
            self.VALIDATION_DIR,
            target_size=(self.width, self.height),
            shuffle=True, 
            color_mode="rgb",
            batch_size=batch_size//2,            
            subset='validation',
            class_mode='binary'
        )

        print(train_generator.class_indices)
        

        reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.4,
            patience=6,
            verbose=1
        )
        '''e_stop=tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5, 
            verbose=0,
            restore_best_weights=True
        )'''
        callbacks=[reduce_lr]

        
        history = model.fit(
            train_generator, 
            steps_per_epoch= len(train_generator.filenames) // batch_size,       
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps= len(validation_generator.filenames) // (batch_size) // 2,
            callbacks=callbacks
        )

        self._save_model(model_name, model)

        with open(self.MODEL_DIR + model_name + '/training_result.csv', mode='w') as data_file:
            fields = ['acc', 'val_acc', 'loss', 'val_loss']
            data_writer = csv.writer(data_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(fields)
            for x in range (0, len(history.history['accuracy'])):
                data_writer.writerow([
                    history.history['accuracy'][x],
                    history.history['val_accuracy'][x],
                    history.history['loss'][x],
                    history.history['val_loss'][x]
                ])       
        
               
        return model
        
    
    def predict(self, model, model_name, DIR, verbose):    
        
        while True:        
           
            print('\nDo you want to erase the current Data in Result folder?')     
            print("Select a Option:")
            print("1 - Yes ")
            print("2 - No")
            if verbose:
                try:
                    op = input()   
                    op = int(op)
                except ValueError:      
                    op = 1
               
            else:
                op = 1    
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
        

        predict_datagen = ImageDataGenerator(rescale=1./255)

        predict_generator = predict_datagen.flow_from_directory(
                DIR[0],
                target_size=(self.width, self.height),
                color_mode="rgb",
                shuffle = False,
                class_mode=None,
                batch_size=1)

        len_files = len(predict_generator.filenames)
        predict = model.predict(predict_generator,steps = len_files)
        
        print("Model accuracy: ", np.max(tf.nn.sigmoid(predict)))
    
        with open(self.MODEL_DIR + model_name + '/predict_result.csv', mode='w') as data_file:
            headerList = ['filename', 'output', 'predict', 'assurance']
            data_writer = csv.writer(data_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(headerList)
            for path in DIR[1]:            
                onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                image_count = len(list(onlyfiles))
                print(image_count)
                
                for x in range (0, image_count):                                                 
                    current_path = path + '/' + onlyfiles[x]
                
                    if predict[x] <= 0.5:                        
                        copyfile(current_path, self.RESULT_DIR + '/Good/' + onlyfiles[x])
                        guessing = (0.5 - predict[x])
                                               
                        print(
                            "This image ({filename}) most likely belongs to Good".format(filename = onlyfiles[x]),
                            "with a {:.1f}% percent confidence.".format(float(guessing + 0.5)*100)
                        )                      
                        data_writer.writerow([onlyfiles[x], 0, float(predict[x]), float(guessing + 0.5)*100])
                    else:
                        
                        copyfile(current_path, self.RESULT_DIR + '/Rotten/' + onlyfiles[x])
                        print(
                            "This image ({filename}) most likely belongs to Rotten".format(filename = onlyfiles[x]),
                            "with a {:.1f}% percent confidence.".format(float(predict[x])*100)
                        )                       
                        data_writer.writerow([onlyfiles[x], 1, float(predict[x]), float(predict[x])*100])
                    
                    x += 1

        chart = Chart(model_name)
        chart.confidence_1_chart()
        chart.confidence_2_chart()
        chart.confidence_3_chart()            
        chart.confusion_matrix_1_chart()
        chart.confusion_matrix_2_chart()
           
              
    def load_model(self, model_name):
        if os.path.isfile(self.MODEL_DIR + model_name + '/' + model_name +  '.h5') is True:
            json_file = open(self.MODEL_DIR + model_name + '/' + model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json, custom_objects={'leaky_relu': tf.nn.leaky_relu})
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
                    
        
  