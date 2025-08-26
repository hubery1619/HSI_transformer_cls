import io
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

import landscape_op.meters as meters


@torch.no_grad()
def test(model, n_ff, dataset,
         transform=None, smoothing=0.0,
         bins=np.linspace(0.0, 1.0, 11),
         verbose=False, period=10, gpu=True):
    model.eval()
    model = model.cuda() if gpu else model.cpu()
    xs, ys = next(iter(dataset))
    xs = xs.cuda() if gpu else xs.cpu()
    nll_meter = meters.AverageMeter("nll")
    metrics = None

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()
        if transform is not None:
            xs, ys_t = transform(xs, ys)
        else:
            xs, ys_t = xs, ys
        if len(ys_t.shape) > 1:
            loss_function = SoftTargetCrossEntropy()
            ys = torch.max(ys_t, dim=-1)[1]
        elif smoothing > 0.0:
            loss_function = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda() if gpu else loss_function

        # A. Predict results
        ys_pred = torch.stack([F.softmax(model(xs), dim=1) for _ in range(n_ff)])
        ys_pred = torch.mean(ys_pred, dim=0)

        ys_t = ys_t.cpu()
        ys = ys.cpu()
        ys_pred = ys_pred.cpu()

        # B. Measure Confusion Matrices
        nll_meter.update(loss_function(torch.log(ys_pred), ys_t).numpy())
        nll_value = nll_meter.avg
        metrics = nll_value
        if verbose and int(step + 1) % period == 0:
            print("%d Steps, %s" % (int(step + 1), repr_metrics(metrics)))

    print(repr_metrics(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.tight_layout()
    calibration_image = plot_to_image(fig)
    if not verbose:
        plt.close(fig)

    return metrics


def repr_metrics(metrics):
    nll_value = metrics

    metrics_reprs = [
        "NLL: %.4f" % nll_value if nll_value > 0.01 else "NLL: %.4e" % nll_value,
    ]
    return ", ".join(metrics_reprs)

def save_lists(metrics_dir, metrics_list):
    with open(metrics_dir, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for metrics in metrics_list:
            writer.writerow(metrics)

def save_metrics(metrics_dir, metrics_list):
    metrics_acc = []
    for metrics in metrics_list:
        *keys, \
        nll_value = metrics

        metrics_acc.append([
            *keys,
            nll_value
        ])

    save_lists(metrics_dir, metrics_acc)

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by "figure" to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)

    trans = transforms.ToTensor()
    image = buf.getvalue()
    image = Image.open(io.BytesIO(image))
    image = trans(image)

    return image

