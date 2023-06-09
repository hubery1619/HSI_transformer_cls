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






treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/hu/proposed/hu_proposed_0_x0.1_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x1 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/hu/proposed/hu_proposed_0_x0.2_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x2 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/hu/proposed/hu_proposed_0_x0.4_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x3 = links_update


treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/hu/proposed/hu_proposed_0_x0.6_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x4 = links_update



treatment_spell_path = r'/home/tirgan/a/liu3044/Project/Group_Transformer_hyper/leaderboard/logs/hu/proposed/hu_proposed_0_x0.8_hessian_matrics.csv'
links = []

with open(treatment_spell_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x5 = links_update



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
values1 = asarray([value for value in range(-1500, 1500)])
values1 = values1.reshape((len(values1), 1))
probabilities1 = model.score_samples(values1)
probabilities1 = exp(probabilities1)
# plot the histogram and pdf
# pyplot.hist(sample, bins=50, density=True)
# pyplot.plot(values[:], probabilities)
# pyplot.show()


sample = x2
# fit density
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values2 = asarray([value for value in range(-1500, 1500)])
values2 = values2.reshape((len(values2), 1))
probabilities2 = model.score_samples(values2)
probabilities2 = exp(probabilities2)


sample = x3
# fit density
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values3 = asarray([value for value in range(-1500, 1500)])
values3 = values3.reshape((len(values3), 1))
probabilities3 = model.score_samples(values3)
probabilities3 = exp(probabilities3)



sample = x4
# fit density
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values4 = asarray([value for value in range(-1500, 1500)])
values4 = values4.reshape((len(values4), 1))
probabilities4 = model.score_samples(values4)
probabilities4 = exp(probabilities4)



sample = x5
# fit density
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values5 = asarray([value for value in range(-1500, 1500)])
values5 = values5.reshape((len(values5), 1))
probabilities5 = model.score_samples(values5)
probabilities5 = exp(probabilities5)



fig, ax = plt.subplots(1, 1, figsize=(6.5, 4), dpi=200)
ax.plot(values1[:], probabilities1, label='0.1')
ax.plot(values2[:], probabilities2, label='0.2')
ax.plot(values3[:], probabilities3, label='0.4')
ax.plot(values4[:], probabilities4, label='0.6')
ax.plot(values5[:], probabilities5, label='0.8')


ax.set_xlabel("Values")
ax.set_ylabel("Probability")
ax.legend(labels=['0.1', '0.2', '0.4', '0.6', '0.8'])

save_path = 'output_result/hessian/' + str('hessian') + '_' + 'density_hyper_trans_0.1_50up.png'
fig.savefig(save_path, bbox_inches = 'tight')
# plt.show()












# kde = grid.best_estimator_
# pdf = np.exp(kde.score_samples(x_grid[:, None]))

# fig, ax = plt.subplots()
# ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
# ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
# ax.legend(loc='upper left')
# ax.set_xlim(min(x), max(x))