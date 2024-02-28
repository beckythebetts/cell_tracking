from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import time as pytime

from Cells import Cell
from plot_features import plot_features

reached_end = False
amoeba_index = 1
while not reached_end:
    print('----------\nAMOEBA '+str(amoeba_index)+'\n----------')
    LENGTH = len(os.listdir('amoeba'))
    LENGTH = 5
    cell = Cell(masks=[np.where(plt.imread(Path('amoeba') / str(str(mask) + '.tif')) == amoeba_index, 1, 0) for mask in range(LENGTH)], index=amoeba_index, type='amoeba')
    if not np.any(cell.masks):
        reached_end = True
    for time in range(LENGTH):
        start_time = pytime.time()
        print('FRAME:', time)
        cell.write_features(time)
        print('Time to run frame ', pytime.time()-start_time)
    plot_features(Path('info_amoeba'), str(amoeba_index))
    amoeba_index += 1
