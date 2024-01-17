import numpy as np
import matplotlib.pyplot as plt

# # 创建数据
# x = np.linspace(0, 6*np.pi, 1000)
# y = np.sin(x) + np.random.normal(0, 0.1, 1000)

# # 使用ax.plot()画出曲线
# fig, ax = plt.subplots()
# ax.plot(x, y)

# # 找到y值最大的点
# max_y_index = np.argmax(y)
# max_y_x_val = x[max_y_index]
# print(max_y_x_val)
# # 在该点上绘制平行于x=0的虚线
# ax.axvline(max_y_x_val, color='r', linestyle='--')

# ax.text(max_y_x_val, ax.get_ylim()[0], f"{max_y_x_val:.2f}", color='r', va='bottom', ha='center')
# fig.savefig('peak_plot.png', bbox_inches = 'tight')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.linspace(0, 6*np.pi, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, 1000)

# 使用ax.plot()画出曲线
fig, ax = plt.subplots()
ax.plot(x, y)

# 找到y值最大的点
max_y_index = np.argmax(y)
max_y_x_val = x[max_y_index]
max_y = y[max_y_index]


ax.scatter(max_y_x_val, max_y, color='r', zorder=5)  # zorder确保点在线之上

# 在该点上绘制平行于x=0的虚线
ax.axvline(max_y_x_val, color='r', linestyle='--', ymin=0, ymax=max_y/ax.get_ylim()[1])

# 在横坐标上标注x的值
ax.text(max_y_x_val, ax.get_ylim()[0], f"{max_y_x_val:.2f}", color='r', va='bottom', ha='center')
fig.savefig('peak_plot.png', bbox_inches = 'tight')

plt.show()