A binary image classifier which utilizes Tensorflow as backend and Keras as Convolutional Neural Network

Still need to figure out how to build the best Keras model for this specific database.

USEFULL LINKS:

https://www.tensorflow.org/tutorials/images/classification
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://www.youtube.com/watch?v=VwVg9jCtqaU&t=1s
https://www.youtube.com/watch?v=IubEtS2JAiY&t=690s

ABOUT THE CODE:

- Inside Data Folder there is a Result folder with 2 subfolders named Good and Rotten, and a Augmented folder. 

- To run Tensorflow, you need at least 3 GB of free RAM or a GPU

- Train_Generator no longer saves generated images into augmented folder, this part of the code is commented

TIPS:
- do not use vgg16 with adam optimizer!

- apparently using sigmoid as activation function requires a BatchNormalization() in each layer.

- do not put a Relu activation fuction after a sigmoid in cnn architeture
