import tensorflow as tf
import numpy as np


# Flexible U-net with unrestricted input size

def upsample_and_concat(input):
    x1, x2, output_channels, in_channels = input[0], input[1], input[2], input[3]
    pool_size = 2
    deconv = tf.keras.layers.Conv2DTranspose(output_channels, (2, 2), strides=(2, 2), padding='same')(x1)
    deconv_output = tf.keras.layers.Concatenate(-1)([deconv, x2])
    return deconv_output


def depth_to_space(input):
    return tf.depth_to_space(input[0], input[1])


def Unet():
    input_shape = (None, None, 3)
    input_x = tf.keras.layers.Input(shape=input_shape)

    unet = tf.keras.Sequential()

    Encoder = tf.keras.Sequential()
    Decoder = tf.keras.Sequential()

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_x)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv5)

    up6 = tf.keras.layers.Lambda(upsample_and_concat)([conv5, conv4, 256, 512])
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv6)

    up7 = tf.keras.layers.Lambda(upsample_and_concat)([conv6, conv3, 128, 256])
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv7)

    up8 = tf.keras.layers.Lambda(upsample_and_concat)([conv7, conv2, 64, 128])
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv8)

    up9 = tf.keras.layers.Lambda(upsample_and_concat)([conv8, conv1, 32, 64])
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv9)

    conv10 = tf.keras.layers.Conv2D(3, (1, 1), padding='same')(conv9)
    print(conv10)
    # out = tf.keras.layers.DepthwiseConv2D((1,1),depth_multiplier = float(1/3))(conv10)
    # out = tf.keras.layers.Lambda(lambda x: tf.depth_to_space(x,3))(conv10)

    return tf.keras.models.Model(input_x, conv10)


Unet().summary()
