
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


TRAINING_DIR = "./Data"
PREDICTION_DIR = "./Prediction"


onlyfiles = [f for f in os.listdir(TRAINING_DIR+'/Good') if os.path.isfile(os.path.join(TRAINING_DIR+'/Good', f))]
onlyfiles2 = [f for f in os.listdir(TRAINING_DIR+'/Rotten') if os.path.isfile(os.path.join(TRAINING_DIR+'/Rotten', f))]

image_count = len(list(onlyfiles))
print(image_count)
width = 0
height = 0
batch_size = 32

for x in range (0, image_count):
    image = PIL.Image.open(TRAINING_DIR + '/Good/' + onlyfiles[x])
    
    w, h = image.size
    width += w
    height += h

onlyfiles = onlyfiles + onlyfiles2
image_count = len(list(onlyfiles))

width = int(width/image_count)
height = int(height/image_count)

TRAINING_DIR = pathlib.Path(TRAINING_DIR)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
TRAINING_DIR,
validation_split=0.2,
subset="training",
seed=123,
image_size=(height, width),
batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  TRAINING_DIR,
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

num_classes = 2

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

epochs=11
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

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
plt.show()

onlyfiles = [f for f in os.listdir(PREDICTION_DIR) if os.path.isfile(os.path.join(PREDICTION_DIR, f))]
image_count = len(list(onlyfiles))

for x in range (0, image_count):
    image = PIL.Image.open(PREDICTION_DIR + '/' + onlyfiles[x])
   
    img = keras.preprocessing.image.load_img(
        PREDICTION_DIR + '/' + onlyfiles[x], target_size=(height, width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image ({}) most likely belongs to {} with a {:.2f} percent confidence."
        .format(onlyfiles[x] ,class_names[np.argmax(score)], 100 * np.max(score))
    )

