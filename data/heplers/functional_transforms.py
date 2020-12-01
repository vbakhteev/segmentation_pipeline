import numpy as np
import scipy
import skimage
from scipy import ndimage


def resize_img3d(img: np.ndarray, target_size: tuple):
    scale = get_scale(img, target_size)
    resized_img = scipy.ndimage.zoom(img, scale)
    return resized_img


def resize_mask3d(mask, target_size):
    scale = get_scale(mask, target_size)
    ohe_mask = ohe_ndarray(mask)

    # Process each ohe channel individually
    resized = []
    for ch in range(ohe_mask.shape[-1]):
        resized_ch = ndimage.zoom(ohe_mask[:, :, :, ch], scale)
        new_mask = skimage.img_as_bool(resized_ch)
        resized += [new_mask]
    resized = np.stack(resized).astype(np.uint8)

    # From ohe to indices
    resized = np.argmax(resized, axis=0)
    return resized


def get_scale(img, target_shape):
    img_shape = np.array(img.shape)
    target_shape = np.array(target_shape)

    scale = target_shape / img_shape
    return scale


def ohe_ndarray(array):
    shape = array.shape
    values = array.reshape(-1).astype(np.uint8)
    n_values = np.max(values) + 1
    ohe = np.eye(n_values)[values]

    return ohe.reshape(*shape, n_values)
