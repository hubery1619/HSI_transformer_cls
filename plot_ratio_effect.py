# PU dataset
# x = [0.02, 0.06, 0.1, 0.14, 0.18, 0.22] //2

import matplotlib
from matplotlib import pyplot as plt
x = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
y_cnn = [89.92, 96.37, 98.54, 99.10, 99.29, 99.42]
y_ssa = [88.07, 96.88, 98.48, 98.95, 99.23, 99.46]
y_csa = [87.98, 97.08, 98.50, 98.82, 99.22, 99.42]
y_ssacnn = [88.57, 96.34, 98.64, 98.96, 99.39, 99.44]
y_csacnn = [89.98, 96.42, 98.68, 98.83, 99.38, 99.56]
# plt.plot(x, y_cnn, color='green', marker='o', linestyle='solid')
# plt.plot(x, y_ssa, color='blue', marker='s', linestyle='dashed')
# # plt.title(?)
# plt.xlabel("Training ratio")
# plt.ylabel("Overall accuracy")


fig, ax = plt.subplots(1, 1, figsize=(4.0, 3), dpi=200)
ax.plot(x, y_cnn, color='green', marker='o', linestyle='solid', label='CNN-mixer')
ax.plot(x, y_ssa, color='blue', marker='s', linestyle='dashed', label='SSA-mixer')
ax.plot(x, y_csa, color='red', marker='v', linestyle='dashdot', label='CSA-mixer')
ax.plot(x, y_ssacnn, color='cyan', marker='+', linestyle='dotted', label='SSA+CNN-mixer')
ax.plot(x, y_csacnn, color='black', marker='^', linestyle='dashed', label='CSA+CNN-mixer')
ax.grid(True, linestyle='--')



ax.set_xlabel("Training ratio")
ax.set_ylabel("OA of Houston 2013 dataset (%)")
ax.legend(labels=['CNN-mixer', 'SSA-mixer', 'CSA-mixer', 'SSA+CNN-mixer', 'CSA+CNN-mixer'])

save_path = 'output_result/OA/hu/' + 'OA_training_ratio_hu.png'
# ax.set_xlim(xmin=-1000, xmax=1000)
# ax.set_ylim(ymin=0, ymax=0.004)
fig.savefig(save_path, bbox_inches = 'tight')
