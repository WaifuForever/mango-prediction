import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import json

class Chart:
   
    MODEL_DIR = "./Models/"
    def __init__(self):
       pass
        
    def training_chart(self, model_name):
        try:
            with open( self.MODEL_DIR + model_name + '/' + model_name + '_data.txt') as json_file:
                data = json.load(json_file)
                epochs = range(data['epochs_range'])

                plt.style.use("fivethirtyeight")
                plt.figure(figsize=(8, 8))
                plt.subplot(1, 2, 1)
                plt.plot(epochs, np.array(data['acc']), label='Training Accuracy', marker='o', linewidth=2.0)
                plt.plot(epochs, np.array(data['val_acc']), label='Validation Accuracy', marker='o', linewidth=2.0)
                plt.legend(loc='lower right')
                plt.title('Accuracy')
                

                plt.subplot(1, 2, 2)
                plt.plot(epochs, np.array(data['loss']), label='Training Loss', marker='o', linewidth=2.0)
                plt.plot(epochs, np.array(data['val_loss']), label='Validation Loss', marker='o', linewidth=2.0)
                plt.legend(loc='upper right')
                plt.title('Loss')
                plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_training.png')
                
                plt.show()
        except ValueError as e:
            print(e) 

    def precision_chart(self, model_name, scores):

        plt.style.use("fivethirtyeight")

        x=["Good", "Rotten", "Total"]      
        width=0.25

        x_indexes = np.arange(len(x))
        y_indexes = [scores[0] * 100/scores[2], scores[1] * 100/scores[2]]

        plt.bar(x_indexes, y_indexes, width=width)
                
        plt.legend(loc='lower right')
        plt.title('Precision')
        plt.xticks(ticks=x_indexes, labels=x)

        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name + '_precision' +'.png')    
        plt.show()

    def guessing_chart(self, model_name, scores):

        plt.style.use("fivethirtyeight")
        
        plt.figure(figsize=(8, 8))
               
        plt.plot(range(len(scores["Good"])), scores["Good"], linewidth=1.0)
        plt.plot(range(len(scores["Rotten"])), scores["Rotten"], linewidth=1.0)
        #plt.plot(range(len(scores["Average"])), scores["Average"], label="Average", linewidth=1.0)

        plt.legend(loc='lower right')
        plt.title('Guessing')
       
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name + '_guessing_1' +'.png')    
        plt.show()


        average_g = 0
        average_r = 0

        for y in scores["Good"]:
            average_g += y

        for y in scores["Rotten"]:
            average_r += y

        average_g = average_g/len(scores["Good"])
        average_r = average_r/len(scores["Rotten"])

        dictlist = []
        for key, value in scores.items():
            dictlist.append(key)

       
        x_indexes = np.arange(len(dictlist))
        width=0.5        
        plt.bar(dictlist, [average_g, average_r, (average_g + average_r)/2], width=width,)
       
        plt.xticks(ticks=x_indexes, labels=dictlist)

        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name + '_guessing_2' +'.png')    
        plt.show()


    
    
