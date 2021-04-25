from skimage import io
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
class Data:

    TRAINING_DIR = "./Data/Training"
    VALIDATION_DIR = "./Data/Validation"

  

    def __init__(self):
            pass

    def _count_images(self, path):       
        num_files = 1          
        num_files += len([f for f in os.listdir(path+'/Good')if os.path.isfile(os.path.join(path+'/Good', f))])
        num_files += len([f for f in os.listdir(path+'/Rotten')if os.path.isfile(os.path.join(path+'/Rotten', f))])
        
        print(num_files)
        return num_files
        