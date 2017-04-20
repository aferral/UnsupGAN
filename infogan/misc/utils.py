from __future__ import print_function
from __future__ import absolute_import
import errno
import os
import scipy.misc
import numpy as np
from sklearn import metrics


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def merge(images, size):

    if len(images.shape) == 2:
        h = np.sqrt(images.shape[1]).astype(np.int32)
        w=h
        images = images.reshape((images.shape[0], h, w))
        img = np.zeros((h * size[0], w * size[1]))
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if len(image.shape) == 2:
            img[j*h:j*h+h, i*w:i*w+w] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def get_image(image_path, is_crop=True, resize_w=64):
    return transform(imread(image_path), is_crop, resize_w)


def imread(path):
    im = scipy.misc.imread(path).astype(np.float)
    if im.shape[-1] != 3:
        if im.shape[-1] == 4:  # some imagenet images have 4 channels (weird!)
            im = im[:, :, :3]
        elif len(im.shape) == 2:  # grayscale images
            im3 = np.zeros((im.shape[0], im.shape[1], 3))
            im3[:, :, 0] = im
            im3[:, :, 1] = im
            im3[:, :, 2] = im
            im = im3
    return im


def center_crop(x, resize_w=64):
    h, w = x.shape[:2]
    if h < w:
        crop_x = h
    else:
        crop_x = w
    crop_h = crop_x
    crop_w = crop_x

    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def inverse_transform(images):
    return (images+1.)/2.


def transform(image, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def compute_cluster_scores(labels, pred_labels, path):
    assert len(labels) == len(pred_labels)
    if len(labels.shape) == 2:
	labels = np.where(labels)[1] #FROM ONE HOT TO INTEGER
    rand_score = metrics.adjusted_rand_score(labels, pred_labels)
    nmi_score = metrics.normalized_mutual_info_score(labels, pred_labels)
    with open(path, 'a') as rr:
        rr.write("%4.4f %4.4f\n" % (rand_score, nmi_score))
