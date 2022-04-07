import os

'''
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory('C:/Users/Loska/Desktop/zlib/dll_x64')
'''

# Ignorar información de Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#

import tensorflow as tf
import numpy as np
import datetime
from models import get_model
from keras.preprocessing.image import ImageDataGenerator

def rotate_image(image):
    return np.rot90(image, np.random.choice([0, 1, 2, 3]))

if __name__ == '__main__':

    train_dir = "data_split/train"
    valid_dir = "data_split/valid"
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function = rotate_image
    )

    valid_gen = ImageDataGenerator(
        
    )

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True  
    )

    valid_data = train_gen.flow_from_directory(
        valid_dir,
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )

    class_names = ["flair", "t1", "t2"]

    model = get_model() 
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint('models/model{epoch:02d}.h5', period=1) 
    ]

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=11,
        steps_per_epoch=train_data.samples // batch_size,
        validation_steps=valid_data.samples // batch_size,
        callbacks=callbacks
    )