import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
import matplotlib.pyplot as plt
import os


from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from models.proposed import proposed
from train import train, test
from timm.loss import LabelSmoothingCrossEntropy
import torch.optim as optim


if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="HSI classification")
    parser.add_argument("--model", type=str, default='cnn3d')
    parser.add_argument("--dataset_name", type=str, default="hu")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument("--trans_type", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=5) 
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)  # bs = batch size  
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--disjoint', action='store_false')  # disjoint the training patch and testing patch  

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))    
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    opts.disjoint = False
    training_split = opts.ratio
    print("The model's type: {}".format(opts.trans_type))
    if opts.disjoint:
        print("{} for training with disjoint sampling".format(opts.ratio))
    else:
        print("{} for training, {} for validation and {} testing with random setting".format(opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir, gt_file = "gt.mat", mat_name = 'gt')

    num_classes = len(labels)
    num_bands = image.shape[-1]

    # random seeds
    seeds = [1, 11, 21, 31, 41]
    # empty list to storing results
    results = []

    metric_output_dir = "./outout_metrics/" + opts.model + '/' + opts.dataset_name 

    if opts.model == 'proposed':
        metric_output_filename = str(training_split) + '_' + 'disjoint:' + str(opts.disjoint) + '_' + 'trans_type:' + str(opts.trans_type) + '_' + 'patch_size:' + str(opts.patch_size) + '_metric_output.txt'
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
                # print("network information:")
                with torch.no_grad():
                    summary(model, torch.zeros((3, 1, num_bands, opts.patch_size, opts.patch_size)))
            
            model = model.to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

            if opts.smoothing:
                criterion = LabelSmoothingCrossEntropy(smoothing=opts.smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
            model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(opts.epoch) + '/'  + str(opts.trans_type) + '/' + str(opts.ratio) + '/' + str(run)
            try:
                best_OA_validation, loss_training, loss_validation = train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device)
                x_file.write("Best validation overall accuracy {}".format(best_OA_validation))
                x_file.write('\n') 
            except KeyboardInterrupt:
                print('"ctrl+c" is pused, the training is over')
            metric_output_dir = "./output_result/losscurve/" + opts.dataset_name + '/' + opts.model + '/' + str(opts.ratio)
            metric_output_filename = str(opts.epoch) + '_' + 'disjoint:' + str(opts.disjoint) + '_' + 'trans_type:' + str(opts.trans_type) + '_' + 'patch_size:' + str(opts.patch_size) + str(run) + '_losscurve.png'
            if not os.path.isdir(metric_output_dir):
                os.makedirs(metric_output_dir, exist_ok=True)

            save_path = os.path.join(metric_output_dir, metric_output_filename)

            fig, ax = plt.subplots()
            x = list(range(0, opts.epoch, 1))

            ax.plot(x, loss_training, label='training loss')
            ax.plot(x, loss_validation, label='validation loss')
            ax.set_xlabel('Training epoch')
            ax.set_ylabel('Loss value')
            ax.set_title('Loss curve')
            ax.legend()
            fig.savefig(save_path, bbox_inches = 'tight')
            plt.show()


            # test the model
            probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
            
            prediction = np.argmax(probabilities, axis=-1)

            # computing metrics
            run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
            results.append(run_results)
            metric_result_single = show_results(run_results, label_values=labels)
            x_file.write('{}'.format(metric_result_single))

            del model, train_set, train_loader, val_set, val_loader

        if opts.num_run > 1:
            metric_result_average = show_results(results, label_values=labels, agregated=True)
            x_file.write('{}'.format(metric_result_average))

        


