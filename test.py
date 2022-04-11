import os

os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory('C:/Users/Loska/Desktop/zlib/dll_x64')

# Ignorar información de Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#

import tensorflow as tf
import argparse
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', dest='test_dir_path', type = str, required = True,
		help = 'Ruta del directorio con el conjunto de datos de test.')
	parser.add_argument('-m', dest='model_path', type = str, required = True,
		help = 'Ruta del modelo.')
	return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    test_dir = args.test_dir_path
    model_path = args.model_path
    model = tf.keras.models.load_model(model_path)

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256)
    )

    #print(test_data.class_names)

    output = model.predict(test_data)
    scores = tf.nn.softmax(output * 1e9)
    scores = np.argmax(scores)

    predictions = np.array([])
    labels =  np.array([])
    for x, y in test_data:
        predictions = np.concatenate([predictions, np.argmax(model.call(x), axis = -1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    confusion = tf.math.confusion_matrix(
        labels=labels, 
        predictions=predictions, 
        num_classes=3
    ).numpy()

    model.summary()
    model.evaluate(test_data)
    print("\tFlair\tT1\tT2"+
            "\nFlair\t"+str(confusion[0][0])+"\t"+str(confusion[0][1])+"\t"+str(confusion[0][2])+
            "\nT1\t"+str(confusion[1][0])+"\t"+str(confusion[1][1])+"\t"+str(confusion[1][2])+
            "\nT2\t"+str(confusion[2][0])+"\t"+str(confusion[2][1])+"\t"+str(confusion[2][2]))