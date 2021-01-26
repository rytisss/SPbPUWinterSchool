import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2
import numpy as np

def preprocess_image(image):
    """Cast to greyscale and resize to desired size"""
    # normalize image and make 'tensor-like'
    grey_image_norm = image / 255
    grey_image_norm = np.reshape(grey_image_norm, grey_image_norm.shape + (1,))
    grey_image_norm = np.reshape(grey_image_norm, (1,) + grey_image_norm.shape)
    return grey_image_norm

def create_MobileNetV3Small_model(input_size=(224, 224, 3), weigths_path=None):
    base = tf.keras.applications.MobileNetV3Small(include_top=False,
                                           weights='imagenet',
                                           input_shape=input_size)
    base.trainable = True
    model = tf.keras.Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=Adam(lr=1e-3),
                  metrics=['binary_accuracy'])

    if weigths_path != None:
        model.load_weights(weigths_path)
    return model