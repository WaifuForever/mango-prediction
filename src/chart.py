import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import pandas as pd


from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import PIL


class Chart:
   
    MODEL_DIR = "./Models/"
    def __init__(self):
       pass

    def _training_csv(self):
        with open(self.MODEL_DIR + model_name + '/training_result.csv', 'r') as read_obj:
            try:                               
                data = pd.read_csv(read_obj)   
                return data

            except ValueError as e:
                print(e) 
                return None


    def training_1_chart(self, model_name):    
        data = self._training_csv()  
        epochs = range(len(data['acc']))

        plt.style.use("fivethirtyeight")        
        plt.plot(epochs, np.array(data['acc']), label='Training Accuracy', marker='o', linewidth=2.0)
        plt.plot(epochs, np.array(data['val_acc']), label='Validation Accuracy', marker='o', linewidth=2.0)
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_training_1.png')        


    def training_2_chart(self, model_name):    
        data = self._training_csv()  
        epochs = range(len(data['acc']))
        plt.style.use("fivethirtyeight")  

        plt.plot(epochs, np.array(data['loss']), label='Training Loss', marker='o', linewidth=2.0)
        plt.plot(epochs, np.array(data['val_loss']), label='Validation Loss', marker='o', linewidth=2.0)
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_training_2.png')        
        

    def precision_chart(self, model_name):
        try:
            with open(self.TRAINING_DIR + '/validation_result.csv', mode='w') as val_file:
                with open(self.MODEL_DIR + model_name + '/predict_result.csv', 'r') as pred_file:

                    col_list = ["filename", "output"]                

                    pred_data = pd.read_csv(pred_file, usecols=col_list)
                    val_data = pd.read_csv(data_file, usecols=col_list)                             

                
                    y_pred = pred_data.values.tolist()
                    y_true = val_data.values.tolist()
                
                    

                    print(confusion_matrix(y_true, y_pred))
        except ValueError as e:
            print(e) 
            return None, None
                   
        '''
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
        '''


    def _predict_csv(self):
        with open(self.MODEL_DIR + model_name + '/predict_result.csv', 'r') as read_obj:
            try:
                col_list = ["assurance", "output"]                
                data = pd.read_csv(read_obj, usecols=col_list)                             

                good = data[data['output'] == 0]['assurance']      
                rotten = data[data['output'] == 1]['assurance']
               
                good = good.values.tolist()
                rotten = rotten.values.tolist()
                return good, rotten

            except ValueError as e:
                print(e) 
                return None, None


    def assurance_1_chart(self, model_name):        
                good, rotten = self._predict_csv()
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
                            

                plt.plot(range(len(good)), good, label='Assurance', linewidth=1.0)        
                plt.legend(loc='lower right')
                plt.title('Good Assurance')  
                plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_assurance_1.png')  
               

    def assurance_2_chart(self, model_name):
        good, rotten = self._predict_csv()
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


        plt.plot(range(len(rotten)), rotten, label='Assurance', linewidth=1.0)
        plt.legend(loc='lower right')
        plt.title('Rotten Assurance')
        plt.savefig(self.MODEL_DIR + model_name + '/'+ model_name +'_assurance_2.png')
       
       
    def assurance_3_chart(self, model_name):
        good, rotten = self._predict_csv()        
        plt.style.use("fivethirtyeight")        
        average_g = 0
        average_r = 0

        for y in good:
            average_g += y

        for y in rotten:
            average_r += y

        average_g = float(average_g/len(good))
        average_r = float(average_r/len(rotten))

        
        dictlist = ["Good", "Rotten", "Average"]
        
                    
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
            

    def display_chart(self, model_name, chart_name, show_image):
        try:
            im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + chart_name + '.png')
            if show_image:     
                im.show()
                                
        except Exception as e:
            print(e)
            print("The file %s could not be found\n" % (model_name + chart_name))
            exec("""{chart_name}_chart({model})""".format(chart_name=chart_name, model=model_name))
            try:
                im = PIL.Image.open(self.MODEL_DIR + model_name + '/'+ model_name + chart_name + '.png')
                if show_image:      
                    im.show()
                                    
            except Exception as e:
                print(e)
                print("The file %s could not be found\n" % (model_name + chart_name))
                exec("""{chart_name}_chart({model})""".format(chart_name=chart_name, model=model_name))