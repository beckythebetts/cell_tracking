import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from Cells import Cell
import mask_funcs
import config

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
        return mask_funcs.to_masks(str(config.PATH / [x for x in os.listdir(config.PATH)][index]), self.name)

    def initialise(self):
        for mask in mask_funcs.split_mask(self.new_frame(0)):
            self.cells = np.append(self.cells, Cell(masks=mask[np.newaxis, :, :], index=self.max_cell_index()+1))

    def join_new_frame(self, index):
        if index > config.TRACK_CLIP_LENGTH:
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
                if self.cells[i].missing_count < config.TRACK_CLIP_LENGTH and not np.logical_and(old_cell_mask, orig_new_mask>0).any():
                    self.cells[i].masks = np.vstack((self.cells[i].masks, old_cell_mask[np.newaxis, :, :]))
                    self.cells[i].missing_count += 1
                else:
                    self.cells[i].masks = np.vstack((self.cells[i].masks, np.zeros((1, 1024, 1024))))
        for new_cell_mask in mask_funcs.split_mask(new_mask):
            if not np.logical_and(new_cell_mask, self.last_frame()>0).any():
                self.cells = np.append(self.cells, Cell(masks=np.vstack((np.zeros((len(self.cells[0].masks) - 1, 1024, 1024)), new_cell_mask[np.newaxis, :, :])), index=self.max_cell_index()+1))
        mask = np.array([cell.missing_count < config.TRACK_CLIP_LENGTH for cell in self.cells])
        self.cells = self.cells[mask]

    def run_tracking(self):
        self.initialise()
        im = Image.fromarray((self.last_frame()).astype(np.uint16))
        im.save(Path(self.name)/(str(0)+'.tif'))
        for i in range(len(os.listdir(config.PATH))):
            print('FRAME :', i)
            self.join_new_frame(i)
            im = Image.fromarray((self.last_frame()).astype(np.uint16))
            im.save(Path(self.name) / (str(i) + '.tif'))
            if config.VIEW_TRACKS:
                self.show_last_frame(i)


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


amoeba_tracker = Tracker('amoeba')
amoeba_tracker.run_tracking()
yeast_tracker = Tracker('yeast')
yeast_tracker.run_tracking()
