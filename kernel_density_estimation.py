# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
# http://www.sefidian.com/2017/06/14/kernel-density-estimation-kde-in-python/

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
from sklearn.neighbors import KernelDensity
from numpy.random import normal
from numpy import asarray
from numpy import exp


parser = argparse.ArgumentParser(description="Hessian KDE Plot")
parser.add_argument("--hessian_csv_path", type=str, required=True,
                    help="Path to the hessian csv file")
parser.add_argument("--save_path", type=str, default="output_result/hessian/hu/hessian_density.png",
                    help="Path to save the output figure")
args = parser.parse_args()

hessian_csv_path = args.hessian_csv_path
save_path = args.save_path

links = []

with open(hessian_csv_path, 'r') as f:
    reader = csv.reader(f)
    links.extend(reader)

links_update = links[0][1:-1]
links_update = np.array(list(map(eval, links_update)))
x1 = links_update
sample = x1
model = KernelDensity(bandwidth=100, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
values1 = asarray([value for value in range(int(min(x1)), int(max(x1)))])
values1 = values1.reshape((len(values1), 1))
probabilities1 = model.score_samples(values1)
probabilities1 = exp(probabilities1)
fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.5), dpi=200)
ax.plot(values1[:], probabilities1, 'b', label='Pooling-mixer')
max_y_index = np.argmax(probabilities1)
max_y_x_val = values1[max_y_index][0]
max_y = probabilities1[max_y_index]
ax.scatter(max_y_x_val, max_y, color='r', s=5, marker='o', zorder=3)  
ax.plot([max_y_x_val, max_y_x_val], [0, max_y], 'r--')
offset = max_y
ax.text(max_y_x_val+50, 0, f"{round(max_y_x_val)}", color='black', va='bottom', ha='left')
ax.grid(True, linestyle='--')
ax.set_xlabel("Values")
ax.set_ylabel("Distribution of the largest eigenvalue")
save_path = 'output_result/hessian/hu/' + str('hessian') + '_' + 'density_vision_transformer_HSI.png'
ax.set_xlim(xmin=-1000, xmax=1000)
ax.set_ylim(ymin=0, ymax=0.004)
fig.savefig(save_path, bbox_inches = 'tight')