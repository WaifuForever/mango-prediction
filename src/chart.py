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
        
    def train_chart(self, model_name):
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
                plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'.png')
                
                plt.show()
        except ValueError as e:
            print(e) 

    def hits_chart(self, model_name, scores):

        plt.style.use("fivethirtyeight")

        x=["Good", "Rotten", "Total"]      

        plt.bar(x, scores)
        plt.show()
        '''
        self.MODEL_DIR = "./Models/"
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name + '_result' +'.png')    
        plt.show()'''

    def precision_chart(self, model_name, scores):

        plt.style.use("fivethirtyeight")

        x=["Good", "Rotten", "Total"]      

        plt.bar(x, scores)
        plt.show()
        '''
        self.MODEL_DIR = "./Models/"
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name + '_result' +'.png')    
        plt.show()'''


    
    
