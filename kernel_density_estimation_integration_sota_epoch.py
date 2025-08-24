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






treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/sa/proposed/sa_proposed_transtype8_trainingepoch10_trainingratio0.1_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x1 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/sa/proposed/sa_proposed_transtype8_trainingepoch100_trainingratio0.1_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x2 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/sa/proposed/sa_proposed_transtype8_trainingepoch200_trainingratio0.1_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x3 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/sa/proposed/sa_proposed_transtype8_trainingepoch300_trainingratio0.1_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x4 = links_update




bw = 150



from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
# generate a sample
sample = x1
# fit density
model = KernelDensity(bandwidth=bw, kernel='gaussian')
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


sample = x2
# fit density
model = KernelDensity(bandwidth=bw, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values2 = asarray([value for value in range(int(min(x2)), int(max(x2)))])
values2 = values2.reshape((len(values2), 1))
probabilities2 = model.score_samples(values2)
probabilities2 = exp(probabilities2)


sample = x3
# fit density
model = KernelDensity(bandwidth=bw, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values3 = asarray([value for value in range(int(min(x3)), int(max(x3)))])
values3 = values3.reshape((len(values3), 1))
probabilities3 = model.score_samples(values3)
probabilities3 = exp(probabilities3)



sample = x4
# fit density
model = KernelDensity(bandwidth=bw, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values4 = asarray([value for value in range(int(min(x4)), int(max(x4)))])
values4 = values4.reshape((len(values4), 1))
probabilities4 = model.score_samples(values4)
probabilities4 = exp(probabilities4)




fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), dpi=200)
ax.plot(values1[:], probabilities1, label='epoch10')
# ax.plot(values2[:], probabilities2, label='epoch50')
ax.plot(values2[:], probabilities2, label='epoch100')
# ax.plot(values4[:], probabilities4, label='epoch150')
ax.plot(values3[:], probabilities3, label='epoch200')
# ax.plot(values6[:], probabilities6, label='epoch250')
ax.plot(values4[:], probabilities4, label='epoch300')
# ax.plot(values5[:], probabilities5, label='epoch250')


ax.set_xlabel("Values")
ax.set_ylabel("Probability of Proposed CSA+CNN-mixer")
ax.legend(labels=['epoch10', 'epoch100', 'epoch200', 'epoch300'])

save_path = 'output_result/hessian/salinas/' + str('hessian') + '_' + 'density_hyper_trans_0.1ratio_100sota_salinas_CSA+CNN-mixer.png'
fig.savefig(save_path, bbox_inches = 'tight')
# plt.show()












# kde = grid.best_estimator_
# pdf = np.exp(kde.score_samples(x_grid[:, None]))

# fig, ax = plt.subplots()
# ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
# ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
# ax.legend(loc='upper left')
# ax.set_xlim(min(x), max(x))