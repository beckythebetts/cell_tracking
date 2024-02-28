import os
from pathlib import Path
import shutil

ims_dir = 'Test_data/ims'
mask_dir = 'Test_data/masks'

output_ims_dir = 'Test_data/testdata2D/01'
output_mask_dir = 'Test_data/testdata2D/01_GT/SEG'
output_mask_dir_tra = 'Test_data/testdata2D/01_GT/TRA'


all_masks = os.listdir(mask_dir)

for i, mask in enumerate(all_masks):
    shutil.copyfile(os.path.join(mask_dir, mask), os.path.join(output_mask_dir, 'man_seg' + str(i) + '.tif'))
    shutil.copyfile(os.path.join(mask_dir, mask), os.path.join(output_mask_dir_tra, 'man_track' + str(i) + '.tif'))
    shutil.copyfile(os.path.join(ims_dir, mask[4:]), os.path.join(output_ims_dir, 't' + str(i) + '.tif'))

