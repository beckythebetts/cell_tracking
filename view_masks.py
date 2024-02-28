import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)
def save_mask(mask_tiff, im_tiff, save_path):
    mask, im = plt.imread(mask_tiff), plt.imread(im_tiff)
    color_masks = np.zeros(np.append(np.array(np.shape(mask)), 3))
    plt.axis('off')

    split_masks =[np.where(mask == i+1, 1, 0) for i in range(0, np.max(mask)) if i in mask]
    for single_mask in split_masks:
        for color_mask in color_masks.T:
            color_mask += np.random.uniform(low=0.1, high=1)*single_mask.T
    plt.imshow(color_masks)
    #plt.show()

    im = min_max_scale(im).T

    im_RGB = np.stack((im, im, im), axis=0).T

    mask_and_image = ((color_masks*0.3 + im_RGB*0.7)*255).astype(np.uint8)
    image = Image.fromarray(mask_and_image)
    image.save(save_path)

save_mask('challenge_data/20x_withyeast_2D/RES_th_cell0.256_th_seed0.274/mask000.tif', 'challenge_data/20x_withyeast_2D/raw_ims/t000.tif', 'challenge_data/20x_withyeast_2Dpplied_mask.tif')