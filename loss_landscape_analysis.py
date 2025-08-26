import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
import os
import torch
import argparse
import seaborn as sns
import numpy as np
from utils.dataset import load_mat_hsi
from models.proposed import proposed
from train import test
from utils.utils import metrics, show_results
import imageio
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from train import train, test
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from tqdm import tqdm
from pathlib import Path
import landscape_op.tests as tests
import landscape_op.loss_landscapes as lls
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI Hessian matrics")
    parser.add_argument("--weights", type=str, default="./checkpoints/ssftt/hu/0")
    parser.add_argument("--outputs", type=str, default="./results")
    parser.add_argument("--model", type=str, default='cnn3d')
    parser.add_argument("--dataset_name", type=str, default="hu")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--trans_type", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=1) 
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)  # bs = batch size  
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--disjoint', action='store_false')  # disjoint the training patch and testing patch  

    opts = parser.parse_args()
    print("model = {}".format(opts.model))    
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    opts.disjoint = False
    training_split = opts.ratio
    if opts.disjoint:
        print("{} for training with disjoint sampling".format(opts.ratio))
    else:
        print("{} for training, {} for validation and {} testing with random setting".format(opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir, gt_file = "gt.mat", mat_name = 'gt')
    num_classes_la = len(labels)
    num_bands = image.shape[-1]
    transform = Mixup(num_classes=num_classes_la, mixup_alpha=1.0, cutmix_alpha=0.8, prob=1.0, label_smoothing=opts.smoothing)

    seeds = [1, 11, 21, 31, 41]

    # empty list to storing results
    results = []
    for run in range(opts.num_run):
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run], disjoint=opts.disjoint, window_size=opts.patch_size//2)
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run], disjoint=opts.disjoint, window_size=opts.patch_size//2)
        del trainval_gt
        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=True, shuffle=False)
        # load model and loss
        model = proposed(opts.dataset_name, opts.patch_size, opts.trans_type)
        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)
        model.load_state_dict(torch.load(os.path.join(opts.weights, str(opts.epoch), str(opts.trans_type), str(opts.ratio), str(run), 'model_best.pth')))
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(map_location)
        dataset_name = opts.dataset_name 
        uid = "hyper_transformer"
        model_name = opts.model
        start_time = time.time()
        scale = 1e-0
        n = 21
        gpu = torch.cuda.is_available()
        dataset_train = train_loader
        metrics_grid = lls.get_loss_landscape(
            model, 1, dataset_train, transform=None,
            kws=["pos_embed", "relative_position"],
            x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=gpu, smoothing_value=opts.smoothing
        )
        leaderboard_path = os.path.join("leaderboard", "logs", dataset_name, model_name)
        Path(leaderboard_path).mkdir(parents=True, exist_ok=True)
        metrics_dir = os.path.join(leaderboard_path, "%s_%s_%s_x%s_losslandscape.csv" % (dataset_name, model_name, uid, int(1 / scale)))
        metrics_list = [[*grid, *metrics] for grid, metrics in metrics_grid.items()]
        tests.save_metrics(metrics_dir, metrics_list)
        end_time = time.time()
        time_consume = end_time-start_time
        print("Calulation time：%s" %time_consume)
        weight_decay=0.0001
        # load losslandscape raw data of ResNet-50 or ViT-Ti
        names = ["x", "y", "l1", "l2", "NLL"]
        path = metrics_dir
        data = pd.read_csv(path, names=names)
        data["loss"] = data["NLL"] + weight_decay * data["l2"]  # NLL + l2
        # prepare data
        p = int(math.sqrt(len(data)))
        shape = [p, p]
        xs = data["x"].to_numpy().reshape(shape) 
        ys = data["y"].to_numpy().reshape(shape)
        zs0 = data["loss"].to_numpy().reshape(shape)

        if run == 0:
            zs = zs0
        else:
            zs = zs + zs0

    zs = zs - zs[np.isfinite(zs)].min()
    zs[zs > 100] = np.nan

    zs = zs/(opts.num_run + 0.0)

    norm = plt.Normalize(zs[np.isfinite(zs)].min(), zs[np.isfinite(zs)].max())  # normalize to [0,1]
    colors = cm.plasma(norm(zs))
    rcount, ccount, _ = colors.shape

    fig = plt.figure(figsize=(4.0, 3.5), dpi=200)
    ax = fig.add_subplot(projection='3d')
    ax.grid()
    ax.view_init(elev=15, azim=15)  # angle

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    surf = ax.plot_surface(
        xs, ys, zs, 
        rcount=rcount, ccount=ccount,
        facecolors=colors, shade=False,
    )
    surf.set_facecolor((0,0,0,0))

    # remove white spaces
    ax.set_xlabel('Weight in x')
    ax.set_ylabel('Weight in y')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'Loss value', rotation=90)
    adjust_lim = 1 #0.8
    ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
    ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
    ax.set_zlim(0, 20)
    plt.xticks(np.arange(-1, 1.1, 0.5))
    plt.yticks(np.arange(-1, 1.1, 0.5))
    # fig.subplots_adjust(left=0, right=0, bottom=0, top=0)
    ax.axis('on')

    metric_output_dir = "./output_result/landscape/" + opts.dataset_name + '/' + opts.model + '/' + str(opts.ratio)
    metric_output_filename = str(opts.epoch) + '_' + 'disjoint:' + str(opts.disjoint) + '_' + 'trans_type:' + str(opts.trans_type) + '_losslandscape.png'
    if not os.path.isdir(metric_output_dir):
        os.makedirs(metric_output_dir, exist_ok=True)

    save_path = os.path.join(metric_output_dir, metric_output_filename)
    fig.savefig(save_path, bbox_inches = 'tight')
    plt.show()










