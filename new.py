from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\KHERLEELU\Desktop\projects\python\potato',
    class_mode='binary'
)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers

IMAGE_SIZE=256
BATCH_SIZE=32

image_dataset=tf.keras.preprocessing.image_dataset_from_directory(
   'potato',
   shuffle=True,
   image_size=(IMAGE_SIZE,IMAGE_SIZE),
   batch_size=(BATCH_SIZE)
)
class_names=image_dataset.class_names
print(class_names)
