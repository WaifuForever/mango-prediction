import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import pandas as pd
import itertools

from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import PIL


class Chart:
   
    MODEL_DIR = "./Models/"
    def __init__(self, model_name):
        self.model_name = model_name
        pass


    def _validation_csv(self):

        TRAINING_DIR = "./Data/Training"
       
        with open(TRAINING_DIR + '/validation_result.csv') as val_file:
            with open(self.MODEL_DIR + self.model_name + '/predict_result.csv') as pred_file:

                col_list = ["filename", "output"]                

                pred_data = pd.read_csv(pred_file, usecols=col_list)
                val_data = pd.read_csv(val_file, usecols=col_list)                             

            
                y_pred = pred_data.values.tolist()
                y_val = val_data.values.tolist()
                return y_pred, y_val

    def _training_csv(self):
        with open(self.MODEL_DIR + self.model_name + '/training_result.csv', 'r') as read_obj:
            try:                               
                data = pd.read_csv(read_obj)   
                return data

            except ValueError as e:
                print(e) 
                return None

    def _predict_csv(self):
        with open(self.MODEL_DIR + self.model_name + '/predict_result.csv', 'r') as read_obj:
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


    def training_1_chart(self):    
        data = self._training_csv()  
        epochs = range(len(data['acc']))

        plt.style.use("fivethirtyeight")        
        plt.plot(epochs, np.array(data['acc']), label='Training Accuracy', marker='o', linewidth=2.0)
        plt.plot(epochs, np.array(data['val_acc']), label='Validation Accuracy', marker='o', linewidth=2.0)
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_training_1.png')   
        plt.clf()     


    def training_2_chart(self):    
        data = self._training_csv()  
        epochs = range(len(data['acc']))
        plt.style.use("fivethirtyeight")  

        plt.plot(epochs, np.array(data['loss']), label='Training Loss', marker='o', linewidth=2.0)
        plt.plot(epochs, np.array(data['val_loss']), label='Validation Loss', marker='o', linewidth=2.0)
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_training_2.png')  
        plt.clf()      


    def confusion_matrix_1_chart(self):        
        y_pred, y_val = self._validation_csv()
       
        result = []

        for x in y_pred:                   
            result.append(x[1])

        validation = []
        for y in y_val:
            validation.append(y[1])

        cm = confusion_matrix(validation, result)
        
        classes = ["Good", 'Rotten']
        normalize = False

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confused Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_confusion_matrix_1.png')  
        plt.clf()   


    def confusion_matrix_2_chart(self):
        y_pred, y_val = self._validation_csv()

        hits_g = hits_r = total_data = total_good = total_rotten = 0
        for filename in y_pred:
            total_data += 1
            for val_name in y_val:                
                if filename[0] == val_name[0]:
                    if filename[1] == val_name[1]:
                        if val_name[1] == 0:
                            hits_g+=1                            
                        else:
                            hits_r+=1
                

        for val_name in y_val:
            if val_name[1] == 0:
                total_good+=1
            else:
                total_rotten+=1


        plt.style.use("fivethirtyeight")

        x=["Good", "Rotten", "Average"]      
        width=0.25

        x_indexes = np.arange(len(x))
            

        y_indexes = [
            hits_g * 100/total_good,
            hits_r * 100/total_rotten,
            (hits_g * 100/total_good + hits_r * 100/total_rotten)/2
        ]

        plt.bar(x_indexes, y_indexes, width=width)
                
        plt.legend(loc='lower right')
        plt.title('Precision')
        plt.xticks(ticks=x_indexes, labels=x)

        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_confusion_matrix_2.png')  
        plt.clf()   


    def assurance_1_chart(self):        
        good, rotten = self._predict_csv()
        plt.style.use("fivethirtyeight")       
                    
        plt.plot(range(len(good)), good, label='Assurance', linewidth=1.0)        
        plt.legend(loc='lower right')
        plt.title('Good Assurance')  
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_assurance_1.png')
        plt.clf()
               

    def assurance_2_chart(self):
        good, rotten = self._predict_csv()
        plt.style.use("fivethirtyeight")
       
        plt.plot(range(len(rotten)), rotten, label='Assurance', linewidth=1.0)
        plt.legend(loc='lower right')
        plt.title('Rotten Assurance')
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name +'_assurance_2.png')
        plt.clf()
       
       
    def assurance_3_chart(self):
        good, rotten = self._predict_csv()        
        plt.style.use("fivethirtyeight")        
        average_g = 0
        average_r = 0

        for y in good:
            average_g += y

        for y in rotten:
            average_r += y
        
        try:
            average_g = float(average_g/len(good))
            average_r = float(average_r/len(rotten))
        except Exception as e:
                print(e) 
                average_r = 0

        
        dictlist = ["Good", "Rotten", "Average"]
        
                    
        x_indexes = np.arange(len(dictlist))

        y_indexes = [
            average_g,
            average_r,
            (average_g + average_r)/2
        ]
        

        plt.bar(x_indexes, y_indexes , width=0.5)       
        plt.xticks(ticks=x_indexes, labels=dictlist)
        plt.title('Assurance Average')
        plt.savefig(self.MODEL_DIR + self.model_name + '/'+ self.model_name + '_assurance_3' +'.png')
        plt.clf() 
            
    
    def display_chart(self, chart_name, show_image):
        try:
            im = PIL.Image.open(self.MODEL_DIR + self.model_name + '/'+ self.model_name + '_' + chart_name + '.png')
            if show_image:     
                im.show()
                plt.clf()
                                
        except Exception as e:
            print(e)
            print("The file %s could not be found\n" % (self.model_name + '_' + chart_name))           
            try:
                exec("""self.{chart_name}_chart()""".format(chart_name=chart_name))
                im = PIL.Image.open(self.MODEL_DIR + self.model_name + '/'+ self.model_name + '_' + chart_name + '.png')
                if show_image:      
                    im.show()
                    plt.clf()
                                    
            except Exception as e:
                print(e)
                print("The file %s could not be generated\n" % (self.model_name + '_' + chart_name))
                