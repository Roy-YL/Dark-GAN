import imageio
import numpy as np
import tensorflow as tf
import os

def patch(gt_img, st_img, ps):
    # output: patch of size ps
    # crop image to 512*512 patches
    H = gt_img.shape[1]
    W = gt_img.shape[2]
    
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    st_patch = st_img[:, yy:yy + ps, xx:xx + ps, :]
    gt_patch = gt_img[:, yy:yy + ps, xx:xx + ps, :]
    
    # random flip or transpose image
    if np.random.randint(2, size=1)[0] == 1:  # random flip
        st_patch = np.flip(st_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
        pass
    if np.random.randint(2, size=1)[0] == 1:
        st_patch = np.flip(st_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
        pass
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        st_patch = np.transpose(st_patch, (0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
        pass
    # st_patch = np.minimum(st_patch, 1.0)

    return gt_patch, st_patch

def gen_patches(gt_img, st_img, ps, bs):
    gt_imgs = [None] * bs
    st_imgs = [None] * bs
    for i in range(bs):
        gt_imgs[i],st_imgs[i] = patch(gt_img, st_img, ps)
        
    return np.concatenate(gt_imgs, axis=0), np.concatenate(st_imgs, axis=0)

def load_patches(gt_dir, st_dir, batch_shape, is_training=False, image_type=0):
    while True:

        images = np.zeros(batch_shape)
        filenames = []
        idx = 0
        batch_size = batch_shape[0]
        patch_size = batch_shape[1]

        # only the training images
        input_fns = tf.gfile.Glob(gt_dir + '0*.jpg')
        
        input_ids = [int(os.path.basename(input_fn)[0:5]) for input_fn in input_fns]
        
        np.random.shuffle(input_ids)
        # one image at a time 
        for img_id in input_ids:
            # with imageio.imread() as img:
            #     pass
            # find a random short image

            st_files = tf.gfile.Glob(st_dir + '%05d_*.jpg' % img_id)
            #print(st_files)
            st_path = st_files[np.random.random_integers(0, len(st_files) - 1)]
            st_fn = os.path.basename(st_path)
            # find the long exposure image
            gt_files = tf.gfile.Glob(gt_dir + '%05d_*.jpg' % img_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # open gt, open st
            st = imageio.imread(st_path)
            gt = imageio.imread(gt_path)
            
            st = np.expand_dims(st, axis=0)
            gt = np.expand_dims(gt, axis=0)

            st = st / 255.0
            gt = gt / 255.0
            
            # randomly generate bs patchs
            gt_imgs, st_imgs = gen_patches(gt, st, patch_size, batch_size)
            yield gt_imgs, st_imgs, '%05d'%img_id
        
        

if __name__ == '__main__':
    batch_shape = [16,512,512,3]
    dir_path = "learning-to-see/dataset/rgb_Sony/"
    patch_reader = load_patches(dir_path+'long/', dir_path+'short/', batch_shape)

    
    day, night, _ = next(patch_reader)
    print(day.shape)
    print(day[0])
