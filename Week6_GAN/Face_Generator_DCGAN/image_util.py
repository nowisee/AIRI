import numpy as np
import scipy.misc

def load_image(path):
    return scipy.misc.imread(path)

def save_image(path, image):
    scipy.misc.imsave(path, image)

def image_merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) == 4:
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    else:
        img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if len(images.shape) == 4:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def save_images(path, images, size):
    merged_image = image_merge(images, size)
    save_image(path, merged_image)

def resize_image(image, size):
    return scipy.misc.imresize(image, size)
