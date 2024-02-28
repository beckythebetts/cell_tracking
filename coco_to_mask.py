from pycocotools.coco import COCO
import os
from matplotlib import image
from pathlib import Path
import numpy as np
from PIL import Image
import random

#img_dir = '20x_test/am_only_jpeg'
ann_file = 'challenge_data/20x_withyeast_2D_all_ims/labels/labels_my-project-name_2024-02-23-06-51-20.json'
output_path = 'challenge_data/20x_withyeast_2D/true'

coco = COCO(ann_file)

img_ids = coco.getImgIds()

for im in img_ids:

    image = coco.loadImgs(ids=im)[0]
    #file_name = image['file_name']
    file_name = '000.tif'

    ann_ids = coco.getAnnIds(imgIds=im)
    instance_index=1
    for i, ann_id in enumerate(ann_ids):
        ann = coco.loadAnns(ids=ann_id)[0]
        if i == 0:
            mask = np.zeros((np.shape(coco.annToMask(ann))))
        mask = np.where(mask==0, mask + coco.annToMask(ann) * (i + 1), mask)
    file_path = os.path.join(output_path, 'true_mask'+ file_name)
    Image.fromarray(mask.astype(np.uint16)).save(file_path)

