# Al final he utilizado generadores ya definidos por Keras en la parte de entrenamiento.

import os

# Ignorar información de Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#

import tensorflow as tf

class CustomDataset(tf.data.Dataset):

    pass