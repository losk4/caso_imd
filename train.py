import os

#os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
#os.add_dll_directory('C:/Users/Loska/Desktop/zlib/dll_x64')

# Ignorar información de Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#

import tensorflow as tf
import numpy as np
import datetime
from models import get_model
from keras.preprocessing.image import ImageDataGenerator

def rotate_image(image):
    image = tf.image.rot90(image, np.random.choice([0, 1, 2, 3]))
    return image

def std_image(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

if __name__ == '__main__':

    train_dir = "data_split/train"
    valid_dir = "data_split/valid"
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function = rotate_image,
        #zca_whitening=True,
        brightness_range=[0.5, 1.5]
    )

    valid_gen = ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function = rotate_image,
        #zca_whitening=True,
        brightness_range=[0.5, 1.5]
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

    '''train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256)
    )

    train_ds = train_ds.map(std_image, num_parallel_calls=tf.data.AUTOTUNE)

    import matplotlib.pyplot as plt
    class_names = ["flair", "t1", "t2"]
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):  
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()

    mean = tf.keras.metrics.Mean()
    for images, labels in train_ds:
        mean.update_state(images)

    print(mean.result().numpy())

    while True:
        continue
    '''
    
    model = get_model() 
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint('models/model.h5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    ]

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=50,
        steps_per_epoch=train_data.samples//batch_size,
        validation_steps=valid_data.samples//batch_size,
        callbacks=callbacks
    )