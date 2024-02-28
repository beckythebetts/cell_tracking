from segmentation.inference.inference import inference_2d_ctc
from pathlib import Path
import argparse
import torch
import warnings
import numpy as np
import pandas as pd
import my_eval_metrics
import scale_match
import shutil
import matplotlib.pyplot as plt
import plot_eval

warnings.filterwarnings("ignore", category=UserWarning)
#train_dir = Path(r'training_data/LiveCell_CTC_format_2D/01/t0.tif')
def eval_model(model, cell_type, th_cell, th_seed, im_size, scale = 1, apply_clahe=False, apply_merging=False, artifact_correction=False):

    parser = argparse.ArgumentParser(description='KIT-Sch-GE 2021 Cell Segmentation - Inference')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--cell_type', '-ct', default='none', nargs='+', help='Cell type(s) to predict')
    parser.add_argument('--fuse_z_seeds', '-fzs', default=False, action='store_true', help='Fuse seeds in axial direction')
    parser.add_argument('--model', '-m', default='none', type=str, help='Name of the model to use')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int, help='Cell amount threshold to apply splitting post-processing (3D)')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    parser.add_argument('--th_cell', '-tc', default=0.07, type=float, help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, type=float, help='Threshold for seeds')
    args = parser.parse_args()

    args.apply_clahe = apply_clahe
    args.apply_merging = apply_merging
    args.artifact_correction = artifact_correction
    args.batch_size = 1
    args.cell_type = cell_type
    args.fuse_z_seeds = False
    args.model = model
    args.multi_gpu = False
    args.n_splitting = 40
    args.save_raw_pred = False
    args.scale = scale
    args.subset = '01'
    args.th_cell = th_cell
    args.th_seed = th_seed

    dir = Path('challenge_data') / args.cell_type
    result_name = str('RES_th_cell' + str(th_cell) + '_th_seed' + str(th_seed))
    for parameter in zip([apply_clahe, apply_merging, artifact_correction], ['_apply_clahe', '_apply_merging', '_artifact_correction']):
        if parameter[0]:
            result_name = result_name + parameter[1]
    model = Path('models') / 'all' / (args.model + '.pth')
    data_path = dir / 'ims'
    result_path = dir / result_name
    result_path.mkdir(exist_ok=True)
    true_path = dir / 'true'
    device = torch.device("cpu")
    batchsize = args.batch_size

    inference_2d_ctc(model, data_path, result_path, device, batchsize, args)
    # for mask in result_path.glob('*'):
    #     scale_match.resize_mask(mask, im_size)
    print('Evaluating...')
    return my_eval_metrics.eval_mask(true_path / 'true_mask000.tif', result_path / 'mask000.tif')

def infer_model(model, cell_type, th_cell, th_seed, scale = 1, apply_clahe=False, apply_merging=False, artifact_correction=False):

    parser = argparse.ArgumentParser(description='KIT-Sch-GE 2021 Cell Segmentation - Inference')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--cell_type', '-ct', default='none', nargs='+', help='Cell type(s) to predict')
    parser.add_argument('--fuse_z_seeds', '-fzs', default=False, action='store_true', help='Fuse seeds in axial direction')
    parser.add_argument('--model', '-m', default='none', type=str, help='Name of the model to use')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int, help='Cell amount threshold to apply splitting post-processing (3D)')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    parser.add_argument('--th_cell', '-tc', default=0.07, type=float, help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, type=float, help='Threshold for seeds')
    args = parser.parse_args()

    args.apply_clahe = apply_clahe
    args.apply_merging = apply_merging
    args.artifact_correction = artifact_correction
    args.batch_size = 1
    args.cell_type = cell_type
    args.fuse_z_seeds = False
    args.model = model
    args.multi_gpu = False
    args.n_splitting = 40
    args.save_raw_pred = False
    args.scale = scale
    args.subset = '01'
    args.th_cell = th_cell
    args.th_seed = th_seed

    dir = Path('challenge_data') / args.cell_type
    result_name = str('RES_th_cell' + str(th_cell) + '_th_seed' + str(th_seed))
    for parameter in zip([apply_clahe, apply_merging, artifact_correction], ['_apply_clahe', '_apply_merging', '_artifact_correction']):
        if parameter[0]:
            result_name = result_name + parameter[1]
    model = Path('models') / 'all' / (args.model + '.pth')
    data_path = dir / 'ims'
    result_path = dir / result_name
    result_path.mkdir(exist_ok=True)
    true_path = dir / 'true'
    device = torch.device("cpu")
    batchsize = args.batch_size

    inference_2d_ctc(model, data_path, result_path, device, batchsize, args)

def my_evaluator(dir, th_cells=np.linspace(0.225, 0.35, 5), th_seeds=np.linspace(0.165, 0.31, 5)):
    if not (dir / 'ims').exists():
        shutil.copytree(dir / 'raw_ims', dir / 'ims')
        im_size = np.array(np.shape(plt.imread([file for file in (dir / 'raw_ims').iterdir()][0])))
        new_im_size = (im_size // scale_match.find_scale_factor(dir / 'true')).astype(int)
        #for image in (dir / 'ims').glob('*'):
            #scale_match.resize(image, new_im_size)
            #scale_match.hist_match(image)
    else:
        im_size = np.array(np.shape(plt.imread([file for file in (dir / 'raw_ims').iterdir()][0])))
    parameters = [[round(th_cell, 3), round(th_seed, 3)] for th_cell in th_cells for th_seed in th_seeds]
    eval_results = [eval_model('LiveCell_CTC_format_2D_GT_01_model_04', '20x_withyeast_2D', param[0], param[1], im_size) for param in parameters]
    eval_df = pd.DataFrame(np.concatenate((parameters, eval_results), axis=1), columns=['th_cell', 'th_seed', 'MIOU', 'F1', 'Precision', 'Accuracy'])
    print(eval_df)
    eval_df.to_csv(dir / 'eval_results.txt', sep='\t', index=False)
    plot_eval.plot(dir / 'eval_results.txt')


#my_evaluator(Path(r'challenge_data/20x_withyeast_2D'))
infer_model('LiveCell_CTC_format_2D_GT_01_model_04', '20x_withyeast_2D', 0.256, 0.274)

