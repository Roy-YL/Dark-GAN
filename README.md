# Dark-GAN

Required software: Python 3.6, Tensorflow, ImageIO, Scipy, numpy

Dataset from the paper *Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018*.

Code for GAN structure and training is in `gan.py`

---

An experimental project of converting dark images to bright images using U-net and GAN.

![](images\Gan_structure.png)

The Generator loss is a weighted sum of the Discriminator loss of generated image and the Mean Square Loss between generated image and bright image.

Possible Improvements:

1. Joint train the object detection system with the image processing system. Provide direct optimization goal for object detection, instead of dark-to-bright transformation.
2. Stabilize the training process. (e.g. modify the GAN loss)
3. Find a better loss for the Discriminator.

Original Dark Image

![](images\165_dark.jpg)

  Reconstructed Image with Bounding Boxes from the dark image above

![](images/165_gan_yolo.png)

Bright Image with Bounding Boxes

![](images\165_result.png)

