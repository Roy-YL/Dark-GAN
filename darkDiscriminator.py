import tensorflow as tf
import numpy as np


# Simple CNN as discriminator

def discriminator():
    model = tf.keras.Sequential(name='DIS')
    img_shape = (512, 512, 3)

    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same"
                                     , activation='relu', input_shape=img_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same"
                                     , activation='relu', input_shape=img_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same"
                                     , activation='relu', input_shape=img_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


Model = discriminator()
Model.summary()
