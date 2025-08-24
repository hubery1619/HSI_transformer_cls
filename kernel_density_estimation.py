# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
# http://www.sefidian.com/2017/06/14/kernel-density-estimation-kde-in-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import sklearn

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
# from statsmodels.nonparametric.kde import KDEUnivariate
# from statsmodels.nonparametric.kernel_density import KDEMultivariate



# def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
#     """Kernel Density Estimation with Scikit-learn"""
#     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#     kde_skl.fit(x[:, np.newaxis])
#     # score_samples() returns the log-likelihood of the samples
#     log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
#     return np.exp(log_pdf)






treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper_patch_tgrs_hu/leaderboard/logs/bot/ssftt/bot_ssftt_transtype8_trainingepoch300_trainingratio0.2_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x1 = links_update


# treatment_spell_path = r'/home/tirgan/a/liu3044/Project/vit_explanability/leaderboard/logs/cifar100/resnet_50/cifar100_resnet_50_691cc9a9e4_x1_hessian_matrics.csv'
# links = []

# with open(treatment_spell_path, 'r') as f:
#     reader = csv.reader(f)
#     links.extend(reader)

# links_update = links[0][1:-1]
# links_update = np.array(list(map(eval, links_update)))
# x2 = links_update






from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
# generate a sample
sample = x1
# fit density
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values1 = asarray([value for value in range(int(min(x1)), int(max(x1)))])
values1 = values1.reshape((len(values1), 1))
probabilities1 = model.score_samples(values1)
probabilities1 = exp(probabilities1)
# plot the histogram and pdf
# pyplot.hist(sample, bins=50, density=True)
# pyplot.plot(values[:], probabilities)
# pyplot.show()


# sample = x2
# # fit density
# model = KernelDensity(bandwidth=5, kernel='gaussian')
# sample = sample.reshape((len(sample), 1))
# model.fit(sample)
# # sample probabilities for a range of outcomes
# values2 = asarray([value for value in range(int(min(x2)), 1500)])
# values2 = values2.reshape((len(values2), 1))
# probabilities2 = model.score_samples(values2)
# probabilities2 = exp(probabilities2)




fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.5), dpi=200)
ax.plot(values1[:], probabilities1, 'b', label='SSFTT')


max_y_index = np.argmax(probabilities1)

max_y_x_val = values1[max_y_index][0]
max_y = probabilities1[max_y_index]
# print(max_y_x_val)
# 在该点上绘制平行于x=0的虚线
ax.scatter(max_y_x_val, max_y, color='r', s=5, marker='o', zorder=3)  # zorder确保点在线之上
# ax.axvline(max_y_x_val, color='r', linestyle='--', ymin=0, ymax=max_y/ax.get_ylim()[1])
ax.plot([max_y_x_val, max_y_x_val], [0, max_y], 'r--')
# ax.text(max_y_x_val, ax.get_ylim()[0], f"{max_y_x_val:.1f}", color='r', va='bottom', ha='center')
ax.text(max_y_x_val+50, 0, f"{round(max_y_x_val)}", color='black', va='bottom', ha='left')



ax.grid(True, linestyle='--')
# ax.plot(values2[:], probabilities2, label='ResNet')


ax.set_xlabel("Values")
ax.set_ylabel("Distribution of the largest eigenvalue")
# ax.legend(labels=['ViT', 'ResNet'])

save_path = 'output_result/hessian/bot/' + str('hessian') + '_' + 'density_SSFTT_0.1_300.png'
ax.set_xlim(xmin=-1000, xmax=1000)
ax.set_ylim(ymin=0, ymax=0.004)
fig.savefig(save_path, bbox_inches = 'tight')
# plt.show()












# kde = grid.best_estimator_
# pdf = np.exp(kde.score_samples(x_grid[:, None]))

# fig, ax = plt.subplots()
# ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
# ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
# ax.legend(loc='upper left')
# ax.set_xlim(min(x), max(x))