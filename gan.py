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


# Reconstruct image by applying transformation to patches

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


# Load images from directory with a generator
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


optimizer = tf.keras.optimizers.Adam(1e-6)

Discriminator = discriminator()

Discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

Generator = Unet()
input_img = tf.keras.layers.Input(shape=[None, None, 3])

gen_img = Generator(input_img)

Discriminator.trainable = False
isvalid = Discriminator(gen_img)

# GAN model
Combined = tf.keras.models.Model(input_img, [isvalid, gen_img])
Combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1, 1000], optimizer=optimizer)
# Combined.summary()


batch_shape = [4, 512, 512, 3]
# night_image_reader = load_images("/work/zq21/train_night_cropped", batch_shape)
# day_image_reader = load_images("/work/zq21/train_day_cropped", batch_shape)
dir_path = "learning-to-see/dataset/rgb_Sony/"
patch_reader = load_patches(dir_path + 'long/', dir_path + 'short/', batch_shape)

valid = np.ones((batch_shape[0], 1))
fake = np.zeros((batch_shape[0], 1))

# print(Discriminator.metrics_names)
# print(Combined.metrics_names)

Generator.load_weights("flex_generator_3.h5")
Discriminator.load_weights("flex_discriminator_3.h5")

num_epochs = 2000
for i in range(num_epochs):
    # input_imgs = next(night_image_reader)[0]
    # target_imgs = next(day_image_reader)[0]

    target_imgs, input_imgs, _ = next(patch_reader)
    fake_imgs = Generator.predict(input_imgs)

    d_loss_real = Discriminator.train_on_batch(target_imgs, valid)
    d_loss_fake = Discriminator.train_on_batch(fake_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    g_loss = Combined.train_on_batch(input_imgs, [valid, target_imgs])

    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f mse: %f]" % (i, d_loss[0], 100 * d_loss[1], g_loss[1], g_loss[2]))

Generator.save_weights("flex_generator_3.h5")
Discriminator.save_weights("flex_discriminator_3.h5")

# Reconstruct demo images
image_reader = load_images("./short_demo_obj", batch_shape=[2, 2848, 4256, 3])

for _ in range(8):
    images, filenames = next(image_reader)
    out_imgs = reconstruct_image(images, Generator, patch_size=1424, step_size=1424)
    for out_img, filename in zip(out_imgs, filenames):
        imageio.imwrite("./short_demo_converted_test/" + filename, np.uint8(out_img * 255.0))
