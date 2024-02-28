import numpy as np
import matplotlib.pyplot as plt
import datetime

# for truth instance, intersection with total pred mask, return index of overlap/s
def find_centre(mask):
    coordinates = np.where(mask==1)
    return [np.average(coordinates[0]), np.average(coordinates[1])]
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def to_bounding_box(mask):
    coordinates = np.argwhere(mask)
    min_coordinates = np.min(coordinates, axis=0)
    max_coordinates = np.max(coordinates, axis=0)
    return np.array([min_coordinates, max_coordinates])

def bounding_box_iou(box_1, box_2):
    min_x = np.min((box_1[1,0], box_2[1,0]))
    max_x = np.max((box_1[0,0], box_2[0,0]))
    min_y = np.min((box_1[1, 1], box_2[1, 1]))
    max_y = np.max((box_1[0, 1], box_2[0, 1]))
    box_1_area = (box_1[1, 0] - box_1[0, 0])*(box_1[1, 1] - box_1[0, 1])
    box_2_area = (box_2[1, 0] - box_2[0, 0])*(box_2[1, 1] - box_2[0, 1])
    if min_x > max_x and min_y > max_y:
        intersection = (min_x - max_x)*(min_y - max_y)
        iou = intersection/(box_1_area + box_2_area - intersection)
    else:
        iou = 0
    return iou

def MIOU(true_masks, pred_masks, threshold=0.5):
    # segmentation score
    split_true_masks = split_mask(true_masks)
    total_ious = 0
    for true_mask in split_true_masks:
        intersection = np.logical_and(true_mask, pred_masks != 0)
        values = np.unique(pred_masks[intersection])
        if len(values) == 1:
            iou = compute_iou(true_mask, np.where(np.equal(pred_masks, values[0]), 1, 0))
            if iou > threshold:
                total_ious += iou
    return total_ious/ len(split_true_masks)

def F1(true_masks, pred_masks, threshold = 0.5):
    # detection score
    true_bbs = [to_bounding_box(mask) for mask in split_mask(true_masks)]
    pred_bbs = [to_bounding_box(mask) for mask in split_mask(pred_masks)]
    tp = sum([1 for pred_bb in pred_bbs for true_bb in true_bbs if bounding_box_iou(true_bb, pred_bb) > threshold])
    if len(pred_bbs) == 0:
        precision = 0
    else:
        precision = tp/len(pred_bbs)
    if len(true_bbs) == 0:
        recall = 0
    else:
        recall = tp/len(true_bbs)
    if precision == 0 and recall == 0:
        f1_result = 0
    else:
        f1_result = 2*(precision*recall)/(precision+recall)
    return f1_result, precision, recall

def split_mask(mask_full):
    masks = [np.where(mask_full == i+1, 1, 0) for i in range(0, np.max(mask_full)) if i+1 in mask_full]
    return masks

def eval_mask(true_mask_im, pred_mask_im, threshold = 0.5):
    print('Reading Masks', datetime.datetime.now().time())
    true_mask, pred_mask = plt.imread(true_mask_im), plt.imread(pred_mask_im)
    print('Calculating MIOU', datetime.datetime.now().time())
    miou = MIOU(true_mask, pred_mask, threshold)
    print('Calculating F1', datetime.datetime.now().time())
    f1, precision, accuracy = F1(true_mask, pred_mask, threshold)
    print('Done', datetime.datetime.now().time())
    print(f'Mean Intersection Over Union: {miou}')
    print(f'F1: {f1}, Precision: {precision}, Accuracy: {accuracy}')
    return [miou, f1, precision, accuracy]


#eval_mask('challenge_data/20x_2D/true/true_mask000.tif', 'challenge_data/20x_2D/RES_th_cell0.28_th_seed0.2_scale1/mask000.tif')