import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from PIL import Image
import gc
import time as pytime
from skimage import measure
from plot_features import plot_features
import pandas as pd
gc.enable()
path = Path(r'tiffs')
threshold = 0.5
track_clip_length = 5
view_tracks = True

class Cell:

    def __init__(self, masks, index, type):
        self.masks = masks
        self.index = index
        self.type = type
        seed = int.from_bytes(os.urandom(4), byteorder='big')
        random.seed(seed)
        self.color = [random.random(), random.random(), random.random()]
        self.missing_count = 0
        self.file = Path('info_'+type) / (str(index)+'.txt')
        with open(self.file, 'w') as f:
            f.write('dist_moved\tarea\tcircularity\toverlap\tdist_nearest\tindex_nearest')

        #self.info = pd.Dataframe(columns=['dist_moved', 'area', 'circularity'])

    def clip_track(self):
        self.masks = self.masks[-track_clip_length:, :, :]

    def write_features(self, time=None):
        time = len(self.masks) if time is None else time
        dist_nearest, index_nearest = self.nearest(time)
        new_row = '\n' + '\t'.join([str(self.dist_moved(time)), str(self.area(time)), str(self.circularity(time)),
                                    str(self.overlap(time)), str(dist_nearest), str(index_nearest)])
        with open(self.file, 'a') as f:
            f.write(new_row)

    def centre(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            return measure.centroid(self.masks[time])

    def dist_moved(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            if time < 1:
                return 0
            else:
                return np.linalg.norm(self.centre(time) - self.centre(time-1))

    def area(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            return np.sum(self.masks[time])

    def circularity(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]):
            return np.nan
        else:
            if measure.perimeter(self.masks[time])==0:
                return 0
            else:
                return 4*np.pi*self.area(time) / (measure.perimeter(self.masks[time])**2)

    def overlap(self, time=None):
        time = len(self.masks) if time is None else time
        if not np.any(self.masks[time]) or not np.any(self.masks[time]):
            return np.nan
        else:
            iou = cal_iou(self.masks[time], self.masks[time-1])
            if iou == 0:
                return np.nan
            else:
                return iou

    def nearest(self, time=None, type='yeast'):
        time = len(self.masks) if time is None else time
        start_time = pytime.time()
        if not np.any(self.masks[time]):
            return np.nan, np.nan

        else:
            frame_to_search = plt.imread(Path(type) / ''.join([str(time), '.tif']))
            distance = 0
            index_of_nearest = None
            while index_of_nearest is None:
                time1 = pytime.time()
                circle_mask = create_circle(self.centre(time)[::-1], distance)
                time2 = pytime.time()
                print('Creating circle ', time2-time1)
                search_mask = np.where(frame_to_search > 0, 1, 0)
                time3 = pytime.time()
                print('Creating search mask ', time3-time2)
                intersection = np.logical_and(circle_mask, search_mask)
                print('Time to find intersection ', pytime.time()-time3)
                if not np.any(intersection):
                    distance += 1
                else:
                    unique_values, counts = np.unique(frame_to_search[intersection], return_counts=True)
                    index_of_nearest = unique_values[np.argmax(counts)]
            time3 = pytime.time()
            return distance, index_of_nearest


class Tracker:

    def __init__(self, name, cells=np.empty(0, dtype=Cell)):
        self.name = name
        self.cells = cells

    def max_cell_index(self):
        if len(self.cells) == 0:
            return 0
        else:
            return np.max([cell.index for cell in self.cells])

    def new_frame(self, index):
        return to_masks(str(path / [x for x in os.listdir(path)][index]), self.name)

    def initialise(self):
        for mask in split_mask(self.new_frame(0)):
            self.cells = np.append(self.cells, Cell(masks=mask[np.newaxis, :, :], index=self.max_cell_index()+1))


    def join_new_frame(self, index):
        if index > track_clip_length:
            for cell in self.cells: cell.clip_track()
        orig_new_mask = self.new_frame(index)
        new_mask = orig_new_mask.copy()
        old_masks = [cell.masks[-1] for cell in self.cells]
        for i, old_cell_mask in enumerate(old_masks):
            intersection = np.logical_and(old_cell_mask, new_mask != 0)
            values = np.unique(new_mask[intersection], return_counts=True)
            if len(values[0]) > 0:
                max_value = values[0][np.argmax(values[1])]
                new_cell_mask = np.where(np.equal(new_mask, max_value), 1.0, 0)
                self.cells[i].masks = np.vstack((self.cells[i].masks, new_cell_mask[np.newaxis, :, :]))
                new_mask = np.where(np.equal(new_mask, max_value), 0, new_mask)
                self.cells[i].missing_count = 0
            else:
                if self.cells[i].missing_count < track_clip_length and not np.logical_and(old_cell_mask, orig_new_mask>0).any():
                    self.cells[i].masks = np.vstack((self.cells[i].masks, old_cell_mask[np.newaxis, :, :]))
                    self.cells[i].missing_count += 1
                else:
                    self.cells[i].masks = np.vstack((self.cells[i].masks, np.zeros((1, 1024, 1024))))
        for new_cell_mask in split_mask(new_mask):
            if not np.logical_and(new_cell_mask, self.last_frame()>0).any():
                self.cells = np.append(self.cells, Cell(masks=np.vstack((np.zeros((len(self.cells[0].masks) - 1, 1024, 1024)), new_cell_mask[np.newaxis, :, :])), index=self.max_cell_index()+1))
        mask = np.array([cell.missing_count < track_clip_length for cell in self.cells])
        self.cells = self.cells[mask]

    def run_tracking(self):
        self.initialise()
        im = Image.fromarray((self.last_frame()).astype(np.uint16))
        im.save(Path(self.name)/(str(0)+'.tif'))
        for i in range(len(os.listdir(path))):
            print('FRAME :', i)
            start_time = time.time()
            self.join_new_frame(i)
            im = Image.fromarray((self.last_frame()).astype(np.uint16))
            im.save(Path(self.name) / (str(i) + '.tif'))
            if view_tracks:
                self.show_last_frame(i)
            execution_time = time.time() - start_time
            print('Time to execute: ', execution_time, 's\n-----------')


    def last_frame(self):
        frame = np.empty((1024, 1024))
        for cell in self.cells:
            frame += cell.masks[-1, :, :]*cell.index
        return frame

    def show_last_frame(self, index):
        frame = np.empty((1024, 1024, 3))
        for cell in self.cells:
            for i, col in enumerate(cell.color):
                frame[:, :, i] += (cell.masks[-1, :, :]*col)
        # plt.imshow(frame)
        # plt.show()
        im = Image.fromarray((frame*255).astype(np.uint8))
        im.save(Path('view_tracks') / self.name/(str(index)+'.tif'))

    def show_mask(self):
        plt.imshow(self.cells[5].masks[0])
        plt.show()


def to_instance_mask(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    separated_cells = np.zeros_like(mask)
    for label in range(1, num_labels):
        separated_cells[labels == label] = label
    return separated_cells


def to_masks(image_path, type):
    mask_vals = {"amoeba": 127, "yeast": 254, "proximity": 255}
    seg_mask = cv2.imread(image_path)
    seg_mask = seg_mask[:, 1024:]
    if type == 'proximity':
        return np.where(seg_mask[:, :, 2] == 255, 1, 0)
    else:
        return to_instance_mask(np.where(seg_mask[:, :, 2] == mask_vals[type], 1, 0))


def split_mask(mask_full):
    masks = [np.where(mask_full == i+1, 1, 0) for i in range(0, np.max(mask_full)) if i+1 in mask_full]
    return masks







# TRACK AMOEBA AND YEAST
# amoeba_tracker = Tracker('amoeba')
# amoeba_tracker.run_tracking()
# yeast_tracker = Tracker('yeast')
# yeast_tracker.run_tracking()

# TEST CELL VARIABLES FUNCTIONS


reached_end = False
amoeba_index = 1
while not reached_end:
    print('----------\nAMOEBA'+str(amoeba_index)+'\n----------')
    LENGTH = len(os.listdir('amoeba'))
    cell = Cell(masks=[np.where(plt.imread(Path('amoeba') / str(str(mask) + '.tif')) == amoeba_index, 1, 0) for mask in range(LENGTH)], index=amoeba_index, type='amoeba')
    if not np.any(cell.masks):
        reached_end = True
    for time in range(LENGTH):
        start_time = pytime.time()
        print('FRAME:', time)
        cell.write_features(time)
        print(pytime.time()-start_time)
    plot_features(Path('info_amoeba'), str(amoeba_index))
    amoeba_index += 1

# test_cell = Cell(masks=[np.where(plt.imread(Path('amoeba') / str(str(mask) + '.tif')) == 100, 1, 0) for mask in range(LENGTH)], index=100, type='amoeba')
# for time in range(LENGTH):
#     print('FRAME :', time)
#     test_cell.write_features(time)


# distances = [test_cell.dist_moved(time) for time in range(LENGTH)]
# areas = [test_cell.area(time) for time in range(LENGTH)]
# circularities = [test_cell.circularity(time) for time in range(LENGTH)]
# overlaps = [test_cell.overlap(time) for time in range(LENGTH)]
# plt.plot(distances)
# plt.show()
# plt.plot(areas)
# plt.show()
# plt.plot(circularities)
# plt.show()
# plt.plot(overlaps)
# plt.show()
# print(distances)
# print(overlaps)
# print(test_cell.dist_moved())
# print(test_cell.area())
# print(test_cell.circularity())

