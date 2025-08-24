# PU dataset
# x = [0.04, 0.12, 0.20, 0.28, 0.36, 0.44] //2

import matplotlib
from matplotlib import pyplot as plt
x = [0.02, 0.06, 0.10, 0.14, 0.18, 0.22]
y_cnn = [90.18, 98.98, 99.40, 99.72, 99.88, 99.85]
y_ssa = [90.02, 98.62, 99.57, 99.65, 99.90, 99.88]
y_csa = [89.11, 98.81, 99.53, 99.72, 99.95, 99.91]
y_ssacnn = [89.71, 98.55, 99.58, 99.68, 99.85, 99.88]
y_csacnn = [87.80, 98.77, 99.49, 99.68, 99.71, 99.86]
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
ax.set_ylabel("OA of Botswana dataset (%)")
ax.legend(labels=['CNN-mixer', 'SSA-mixer', 'CSA-mixer', 'SSA+CNN-mixer', 'CSA+CNN-mixer'])

save_path = 'output_result/OA/bot/' + 'OA_training_ratio_bot.png'
# ax.set_xlim(xmin=-1000, xmax=1000)
# ax.set_ylim(ymin=0, ymax=0.004)
fig.savefig(save_path, bbox_inches = 'tight')
