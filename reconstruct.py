import tensorflow as tf
from autoencoder import AutoEncoder_keras
from darkDiscriminator import discriminator
from flex_unet import Unet
from patcher import load_patches
import os
import numpy as np
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
from time import time
import imageio

print("Packages imported!")


def reconstruct_image(images, model, patch_size=512, step_size=256):
    r, c = images.shape[1], images.shape[2]
    hmoves = int((r - (patch_size - step_size)) / step_size) + 1
    vmoves = int((c - (patch_size - step_size)) / step_size) + 1
    output_image = np.zeros(images.shape)
    repeats = np.zeros(images.shape)

    for i in range(hmoves):
        for j in range(vmoves):
            if i == hmoves - 1 and j == vmoves - 1:
                output_image[:, r - patch_size:r, c - patch_size:c, :] += model.predict(
                    images[:, r - patch_size:r, c - patch_size:c, :])
                repeats[:, r - patch_size:r, c - patch_size:c, :] += 1
                continue
            if i == (hmoves - 1):
                output_image[:, r - patch_size:r, j * step_size:j * step_size + patch_size, :] += model.predict(
                    images[:, r - patch_size:r, j * step_size:j * step_size + patch_size, :])
                repeats[:, r - patch_size:r, j * step_size:j * step_size + patch_size, :] += 1
                continue
            if j == (vmoves - 1):
                output_image[:, i * step_size:i * step_size + patch_size, c - patch_size:c, :] += model.predict(
                    images[:, i * step_size:i * step_size + patch_size, c - patch_size:c, :])
                repeats[:, i * step_size:i * step_size + patch_size, c - patch_size:c, :] += 1
                continue

            output_image[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size,
            :] += model.predict(
                images[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size, :])
            repeats[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size, :] += 1

    return output_image / repeats


def load_images(input_dir, batch_shape, is_training=False):
    while True:
        images = np.zeros(batch_shape)
        filenames = []
        idx = 0
        batch_size = batch_shape[0]
        for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
            with tf.gfile.Open(filepath, "rb") as f:
                image = imread(f, mode='RGB').astype(np.float) / 255.0
            images[idx, :, :, :] = image
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield images, filenames
                filenames = []
                images = np.zeros(batch_shape)
                idx = 0
        if idx > 0:
            yield images, filenames


Generator = Unet()
Generator.load_weights("flex_generator_3.h5")

image_reader = load_images("./short_demo_obj", batch_shape=[2, 2848, 4256, 3])

for images, filenames in image_reader:
    out_imgs = reconstruct_image(images, Generator, patch_size=1424, step_size=1424)
    for out_img, filename in zip(out_imgs, filenames):
        imageio.imwrite("./short_demo_converted_test/" + filename, np.uint8(out_img * 255.0))
