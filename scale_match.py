import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.transform import resize as sk_resize
from skimage.color import rgb2gray
from my_eval_metrics import split_mask
from skimage.exposure import match_histograms


train_dir = Path(r'training_data/LiveCell_CTC_format_2D/01')

def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)


def average_pix_cell(mask):
    mask = plt.imread(mask)
    return np.sum(np.where(mask != 0, 1, 0)) / np.max(mask)


def find_scale_factor(my_ims_dir, train_ims_dir=train_dir):
    training_data_average = np.mean([average_pix_cell(mask) for mask in train_ims_dir.glob('*')])
    my_data_average = np.mean([average_pix_cell(mask) for mask in my_ims_dir.glob('*')])
    return np.sqrt(my_data_average / training_data_average)


def resize(image_path, shape):
    #image = rgb2gray(plt.imread(image_path))
    image = plt.imread(image_path)
    image = sk_resize(image, shape, 3)
    image = min_max_scale(image)
    #Image.fromarray((image*(2**16-1)).astype(np.uint16)).save(image_path)
    Image.fromarray((image*(2**8-1)).astype(np.uint8), mode='L').save(image_path)


def resize_mask(image_path, shape):
    image = rgb2gray(plt.imread(image_path))
    masks = split_mask(image)
    resized_mask = np.zeros(shape)
    for i, mask in enumerate(masks):
        resized_mask += sk_resize(mask, shape, order=0, anti_aliasing=False, preserve_range=True)*(i + 1)
    Image.fromarray((resized_mask).astype(np.uint16)).save(image_path)


def hist_match(image_path, train_im=train_dir / 't0.tif'):
    image = plt.imread(image_path)
    image = min_max_scale(image)
    ref_image = plt.imread(train_im)
    matched = match_histograms(image, ref_image)
    Image.fromarray(matched).save(image_path)

