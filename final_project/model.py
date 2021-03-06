import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np


def preprocess_image(image):
    # normalize image and make 'tensor-like'
    norm_image = image / 255
    norm_image_norm = np.reshape(norm_image, (1,) + norm_image.shape)
    return norm_image_norm


def create_MobileNetV3Small_model(input_size=(224, 224, 3), weigths_path=None):
    base = tf.keras.applications.MobileNetV3Small(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=input_size)
    base.trainable = True
    tf.keras.utils.plot_model(base, to_file='base.png', show_shapes=True)
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
    model.summary()
    if weigths_path != None:
        model.load_weights(weigths_path)
    return model
