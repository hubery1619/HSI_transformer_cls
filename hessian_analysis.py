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
from train import test
from utils.utils import metrics, show_results
import imageio
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from train import train, test
from timm.loss import LabelSmoothingCrossEntropy
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from models.proposed import proposed
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from pyhessian import hessian
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI Hessian matrics")
    parser.add_argument("--weights", type=str, default="./checkpoints/ssftt/hu/0")
    parser.add_argument("--outputs", type=str, default="./results")

    parser.add_argument("--model", type=str, default='cnn3d')
    parser.add_argument("--dataset_name", type=str, default="hu")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    # parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--trans_type", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=5) 
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
    #print("disjoint setting = {}".format(opts.disjoint))
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
    mixup_function =None
    seeds = [1, 11, 21, 31, 41]
    # empty list to storing results
    results = []
    metric_output_dir = "./outout_metrics/" + opts.model + '/' + opts.dataset_name 
    if opts.model == 'proposed':
        metric_output_filename = str(training_split) + '_' + 'disjoint:' + str(opts.disjoint) + '_' + 'trans_type:' + str(opts.trans_type) + '_metric_output_hessian.txt'
    else:
        metric_output_filename = str(training_split) + '_' + 'disjoint:' + str(opts.disjoint) + '_metric_output.txt'
    if not os.path.isdir(metric_output_dir):
        os.makedirs(metric_output_dir, exist_ok=True)
    output_metric_result = os.path.join(metric_output_dir, metric_output_filename)
    with open(output_metric_result, 'w') as x_file:
        for run in range(opts.num_run):
            np.random.seed(seeds[run])
            print("running an experiment with the {} model".format(opts.model))
            print("run {} / {}".format(run+1, opts.num_run))
            x_file.write("run {} / {}".format(run+1, opts.num_run))
            x_file.write('\n')            
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
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            if mixup_function is not None:
                loss_function = SoftTargetCrossEntropy()
            elif opts.smoothing > 0.0:
                loss_function = LabelSmoothingCrossEntropy(smoothing=opts.smoothing)
            else:
                loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.cuda() if torch.cuda.is_available() else loss_function
            criterion = loss_function
            start_time = time.time()
            max_eigens = []  # a list of batch-wise top-k hessian max eigenvalues
            model = model.cuda()
            # i = 0
            dataset_train = train_loader
            weight_decay = 0.0001
            for xs, ys in tqdm(dataset_train):
                if mixup_function is not None:
                    xs, ys = mixup_function(xs, ys)
                hessian_comp = hessian(model, criterion, data=(xs, ys), weight_decay=weight_decay, cuda=True)  # measure hessian max eigenvalues with NLL + L2 on data augmented (`transform`) datasets
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=10)  # collect top-10 hessian eigenvaues by using power-iteration (https://en.wikipedia.org/wiki/Power_iteration)
                max_eigens = max_eigens + top_eigenvalues  # aggregate top-10 max eigenvalues
    end_time = time.time()
    time_consume = end_time-start_time
    print("Calulation time：%s" %time_consume)
    uid = opts.trans_type
    leaderboard_path = os.path.join("leaderboard", "logs", opts.dataset_name, opts.model)
    Path(leaderboard_path).mkdir(parents=True, exist_ok=True)
    metrics_dir = os.path.join(leaderboard_path, "%s_%s_transtype%s_trainingepoch%s_trainingratio%s_hessian_matrics.csv" % (opts.dataset_name, opts.model, uid, opts.epoch, training_split))
    metrics_list = max_eigens
    # tests.save_metrics(metrics_dir, metrics_list)
    f = open(metrics_dir, "w")
    f.write(str(metrics_list))
    f.close()































    # opts = parser.parse_args()

    # device = torch.device("cuda:{}".format(opts.device))

    # print("dataset: {}".format(opts.dataset_name))
    # print("patch size: {}".format(opts.patch_size))
    # print("model: {}".format(opts.model))

    # image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    # num_classes = len(labels)
    # num_bands = image.shape[-1]















    # # load model and weights
    # model = get_model(opts.model, opts.dataset_name, opts.patch_size, opts.trans_type)
    # print('loading weights from %s' % opts.weights + '/model_best.pth')
    # model = model.to(device)
    # model.load_state_dict(torch.load(os.path.join(opts.weights, 'model_best.pth')))


























    # model.eval()
    # # testing model: metric for the whole HSI, including train, val, and test
    # probabilities = test(model, opts.weights, image, opts.patch_size, num_classes, device=device)
    # prediction = np.argmax(probabilities, axis=-1)

    # run_results = metrics(prediction, gt, n_classes=num_classes)

    # # prediction[gt < 0] = -1   # mask the no label points

    # # color results
    # colored_gt = color_results(gt+1, palette)
    # colored_pred = color_results(prediction+1, palette)

    # outfile = os.path.join(opts.outputs, opts.dataset_name,  opts.model)
    # os.makedirs(outfile, exist_ok=True)

    # imageio.imsave(os.path.join(outfile, opts.dataset_name + '_gt.png'), colored_gt)  # eps or png
    # imageio.imsave(os.path.join(outfile, opts.dataset_name+'_' + opts.model + '_out.png'), colored_pred)  # or png

    # show_results(run_results, label_values=labels)
    # del model