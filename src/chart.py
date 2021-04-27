import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import PIL
import json

class Chart:
   
    MODEL_DIR = "./Models/"
    def __init__(self):
       pass

    def display_charts(self, model_name, op):      
        # open method used to open different extension image file
        if op == 2:
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + '_guessing_1.png')      
                im.show()

            except Exception as e:
                print(e)
                print("The file %s could not be found\n" % (model_name + '_guessing_1' + '.png'))
            

        elif op == 3:
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + '_guessing_2.png')      
                im.show()
            except Exception as e:
                print(e)
                print("The file %s could not be found\n" % (model_name + '_guessing_2' + '.png'))
        
            
        elif op == 4:
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + '_training_1.png')    
                im.show()
            except Exception as e:
                print(e)
                self.training_chart(model_name)
            
        elif op == 5:
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + '_training_2.png')    
                im.show()
            except Exception as e:
                print(e)
                self.training_chart(model_name)           
            

        elif op == 6:       
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + '_precision.png')      
                im.show()
            except Exception as e:
                print(e)
                print("The file %s could not be found\n" % (model_name + '_precision' + '.png'))
            
        
        else:
            print("Invalid option!!\n")
          
    def training_chart(self, model_name):
        try:
            with open( self.MODEL_DIR + model_name + '/' + model_name + '_data.txt') as json_file:
                data = json.load(json_file)
                epochs = range(data['epochs_range'])

                plt.style.use("fivethirtyeight")
                #plt.figure(figsize=(8, 8))
                #plt.subplot(1, 2, 1)
                plt.plot(epochs, np.array(data['acc']), label='Training Accuracy', marker='o', linewidth=2.0)
                plt.plot(epochs, np.array(data['val_acc']), label='Validation Accuracy', marker='o', linewidth=2.0)
                plt.legend(loc='lower right')
                plt.title('Accuracy')
                plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_training_1.png')
                plt.show()
                
                #plt.subplot(1, 2, 2)
                plt.plot(epochs, np.array(data['loss']), label='Training Loss', marker='o', linewidth=2.0)
                plt.plot(epochs, np.array(data['val_loss']), label='Validation Loss', marker='o', linewidth=2.0)
                plt.legend(loc='upper right')
                plt.title('Loss')
                plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_training_2.png')
                
                plt.show()
        except ValueError as e:
            print(e) 

    def precision_chart(self, model_name, scores):

        plt.style.use("fivethirtyeight")

        x=["Good", "Rotten", "Average"]      
        width=0.25

        x_indexes = np.arange(len(x))
        
        y_indexes = [
            scores[0] * 100/scores[2],
            scores[1] * 100/scores[2],
            (scores[0] * 100/scores[2] + scores[1] * 100/scores[2])/2
        ]

        plt.bar(x_indexes, y_indexes, width=width)
                
        plt.legend(loc='lower right')
        plt.title('Precision')
        plt.xticks(ticks=x_indexes, labels=x)

        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name + '_precision' +'.png')    
        plt.show()

    def assurance_chart(self, model_name, scores):



        plt.style.use("fivethirtyeight")
        ax = plt.subplot()
      
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.78,
            1.08,
            "scale = [0-50]",
            transform=ax.transAxes,
            fontsize=14,
            
            bbox=props)

        plt.plot(range(len(scores["Good"])), scores["Good"], label='Assurance', linewidth=1.0)        
        plt.legend(loc='lower right')
        plt.title('Good Assurance')  
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_assurance_1.png')  
        plt.show()


        ax = plt.subplot()
      
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.78,
            1.08,
            "scale = [0-50]",
            transform=ax.transAxes,
            fontsize=14,            
            bbox=props)


        plt.plot(range(len(scores["Rotten"])), scores["Rotten"], label='Assurance', linewidth=1.0)
        plt.legend(loc='lower right')
        plt.title('Rotten Assurance')
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_assurance_2.png')
        plt.show()
        #plt.plot(range(len(scores["Average"])), scores["Average"], label="Average", linewidth=1.0)

        


        average_g = 0
        average_r = 0

        for y in scores["Good"]:
            average_g += y

        for y in scores["Rotten"]:
            average_r += y

        average_g = float(average_g/len(scores["Good"]))
        average_r = float(average_r/len(scores["Rotten"]))

        
        dictlist = []
        for key in scores.keys():
            dictlist.append(key)
        
       
        x_indexes = np.arange(len(dictlist))

        y_indexes = [
            average_g,
            average_r,
            (average_g + average_r)/2
        ]
        

        ax = plt.subplot()
      
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.78,
            1.08,
            "scale = [0-50]",
            transform=ax.transAxes,
            fontsize=14,
            
            bbox=props)
             
        plt.bar(x_indexes, y_indexes , width=0.5)       
        plt.xticks(ticks=x_indexes, labels=dictlist)
        plt.title('Assurance Average')
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name + '_assurance_3' +'.png')    
        plt.show()


    
    
