# plt.savefig('output_image_path.png')

########## houston label
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Set the number of colors to 15
# num_classes = 15

# # Generate colors using seaborn
# colors = sns.color_palette("hls", num_classes)

# # Rectangle width and spacing
# rect_width = 0.5
# spacing = 0.7

# # Calculate figure width based on number of classes
# fig_width = num_classes * spacing

# # Create a canvas of appropriate width
# fig, ax = plt.subplots(figsize=(fig_width, rect_width + 0.3))

# # Draw colored rectangles and their labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((i * spacing, 0), rect_width, rect_width, 
#                              linewidth=1, edgecolor='r', facecolor=color)
#     ax.add_patch(rect)
    
#     # Position the label next to the rectangle
#     ax.text(i * spacing + rect_width / 2, rect_width + 0.05, 
#             f"C{i+1}", horizontalalignment='center')

# # Adjust axis limits
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, rect_width + 0.3)  # Provide some space for the label

# # Set background to white
# ax.set_facecolor("white")
# fig.patch.set_facecolor('white')

# # Turn off the axis
# ax.axis("off")

# # Display the figure
# plt.tight_layout()
# plt.show()



# # ########## bot label
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Set the number of colors to 14
# num_classes = 14

# # Generate colors using seaborn
# colors = sns.color_palette("hls", num_classes)

# # Rectangle width and spacing
# rect_width = 0.5
# spacing = 0.7

# # Calculate figure width based on number of classes
# fig_width = num_classes * spacing

# # Create a canvas of appropriate width
# fig, ax = plt.subplots(figsize=(fig_width, rect_width + 0.3))

# # Draw colored rectangles and their labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((i * spacing, 0), rect_width, rect_width, 
#                              linewidth=1, edgecolor='r', facecolor=color)
#     ax.add_patch(rect)
    
#     # Position the label next to the rectangle
#     ax.text(i * spacing + rect_width / 2, rect_width + 0.05, 
#             f"C{i+1}", horizontalalignment='center')

# # Adjust axis limits
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, rect_width + 0.3)  # Provide some space for the label

# # Set background to white
# ax.set_facecolor("white")
# fig.patch.set_facecolor('white')

# # Turn off the axis
# ax.axis("off")

# # Display the figure
# plt.tight_layout()
# plt.show()



########## pu label
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Set the number of colors to 9
# num_classes = 9

# # Generate colors using seaborn
# colors = sns.color_palette("hls", num_classes)

# # Rectangle width and spacing
# rect_width = 0.5
# spacing = 0.7

# # Calculate figure width based on number of classes
# fig_width = num_classes * spacing

# # Create a canvas of appropriate width
# fig, ax = plt.subplots(figsize=(fig_width, rect_width + 0.3))

# # Draw colored rectangles and their labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((i * spacing, 0), rect_width, rect_width, 
#                              linewidth=1, edgecolor='r', facecolor=color)
#     ax.add_patch(rect)
    
#     # Position the label next to the rectangle
#     ax.text(i * spacing + rect_width / 2, rect_width + 0.05, 
#             f"C{i+1}", horizontalalignment='center')

# # Adjust axis limits
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, rect_width + 0.3)  # Provide some space for the label

# # Set background to white
# ax.set_facecolor("white")
# fig.patch.set_facecolor('white')

# # Turn off the axis
# ax.axis("off")

# # Display the figure
# plt.tight_layout()
# plt.show()


# ############# Houston 2013 with one column
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 15

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = ['Healthy\ngrass', 'Stressed\ngrass', 'Tree', 'Synthetic\ngrass', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking\nlot 1', 'Parking\nlot 2', 'Tennis\ncourt', 'Running\ntrack']

# # 计算最大标签的宽度和高度
# max_label_width = max([max([len(word) for word in label.split("\n")]) for label in labels]) * 0.09
# max_label_height = len(max(labels, key=lambda s: len(s.split("\n"))).split("\n")) * 0.16  # 更新了标签高度

# # 设置画布大小
# fig, ax = plt.subplots(figsize=(max_label_width * num_classes, max_label_height + 0.2))

# # 计算方块位置
# x_positions = [max_label_width * (index + 0.5) for index in range(num_classes)]

# # 绘制方块和标签
# for x, label, color in zip(x_positions, labels, colors):
#     ax.add_patch(mpatches.Rectangle((x - max_label_width / 2, 0.05), max_label_width, max_label_height, color=color))
#     ax.text(x, max_label_height/2 + 0.05, label, ha='center', va='center', color='black', fontsize=10)

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_xlim(0, max_label_width * num_classes)
# ax.set_ylim(0, max_label_height + 0.1)

# plt.tight_layout(pad=0)
# plt.show()
# plt.savefig('label_hu.png', dpi=300)





# ### botswana dataset

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 14

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Water", 
#     "Hippo\ngrass",
#     "Floodplain\ngrasses 1", 
#     "Floodplain\ngrasses 2",
#     "Reeds", 
#     "Riparian", 
#     "Firescar", 
#     "Island\ninterior",
#     "Acacia\nwoodlands", 
#     "Acacia\nshrublands",
#     "Acacia\ngrasslands", 
#     "Short\nmopane", 
#     "Mixed\nmopane",
#     "Exposed\nsoils"
# ]

# # 计算最大标签的宽度和高度
# max_label_width = max([max([len(word) for word in label.split("\n")]) for label in labels]) * 0.09
# max_label_height = len(max(labels, key=lambda s: len(s.split("\n"))).split("\n")) * 0.16

# # 设置画布大小
# fig, ax = plt.subplots(figsize=(max_label_width * num_classes, max_label_height + 0.2))

# # 计算方块位置
# x_positions = [max_label_width * (index + 0.5) for index in range(num_classes)]

# # 绘制方块和标签
# for x, label, color in zip(x_positions, labels, colors):
#     ax.add_patch(mpatches.Rectangle((x - max_label_width / 2, 0.05), max_label_width, max_label_height, color=color))
#     ax.text(x, max_label_height/2 + 0.05, label, ha='center', va='center', color='black', fontsize=10)

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_xlim(0, max_label_width * num_classes)
# ax.set_ylim(0, max_label_height + 0.1)

# plt.tight_layout(pad=0)
# plt.show()
# plt.savefig('label_bot.png', dpi=300)


##pavia university dataset

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 9

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Asphalt",
#     "Meadows",
#     "Gravel",
#     "Trees",
#     "Painted\nmetal sheets",
#     "Bare Soil",
#     "Bitumen",
#     "Self-Blocking\nBricks",
#     "Shadows"
# ]

# # 计算最大标签的宽度和高度
# max_label_width = max([max([len(word) for word in label.split("\n")]) for label in labels]) * 0.09
# max_label_height = len(max(labels, key=lambda s: len(s.split("\n"))).split("\n")) * 0.16

# # 设置画布大小
# fig, ax = plt.subplots(figsize=(max_label_width * num_classes, max_label_height + 0.2))

# # 计算方块位置
# x_positions = [max_label_width * (index + 0.5) for index in range(num_classes)]

# # 绘制方块和标签
# for x, label, color in zip(x_positions, labels, colors):
#     ax.add_patch(mpatches.Rectangle((x - max_label_width / 2, 0.05), max_label_width, max_label_height, color=color))
#     ax.text(x, max_label_height/2 + 0.05, label, ha='center', va='center', color='black', fontsize=10)

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_xlim(0, max_label_width * num_classes)
# ax.set_ylim(0, max_label_height + 0.1)

# plt.tight_layout(pad=0)
# plt.show()


# plt.savefig('label_pu.png', dpi=300)



####自适应代码与论文中数据集表示一致
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 9

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Asphalt",
#     "Meadows",
#     "Gravel",
#     "Trees",
#     "Painted\nmetal sheets",
#     "Bare Soil",
#     "Bitumen",
#     "Self-Blocking\nBricks",
#     "Shadows"
# ]

# # 计算最大标签的宽度和高度
# max_label_width = max([max([len(word) for word in label.split("\n")]) for label in labels]) * 0.09
# max_label_height = len(max(labels, key=lambda s: len(s.split("\n"))).split("\n")) * 0.16

# # 设置画布大小（自适应最长文字的宽度）
# fig_width = max_label_width + 0.0  # 增加一点边缘空间
# fig, ax = plt.subplots(figsize=(fig_width, max_label_height * num_classes))

# # 计算方块位置（竖排的位置）
# y_positions = [max_label_height * (num_classes - index - 0.5) for index in range(num_classes)]

# # 绘制方块和标签
# for y, label, color in zip(y_positions, labels, colors):
#     ax.add_patch(mpatches.Rectangle((0.0, y - max_label_height / 2), max_label_width, max_label_height, color=color))
#     ax.text(max_label_width/2 + 0.0, y, label, ha='center', va='center', color='black', fontsize=10)

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_ylim(0, max_label_height * num_classes)
# ax.set_xlim(0, fig_width)

# plt.tight_layout(pad=0)
# plt.show()

# # 保存图片
# plt.savefig('label_pu_vertical_adapted.png', dpi=300)




### botswana dataset
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 14

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Water", 
#     "Hippo\ngrass",
#     "Floodplain\ngrasses 1", 
#     "Floodplain\ngrasses 2",
#     "Reeds", 
#     "Riparian", 
#     "Firescar", 
#     "Island\ninterior",
#     "Acacia\nwoodlands", 
#     "Acacia\nshrublands",
#     "Acacia\ngrasslands", 
#     "Short\nmopane", 
#     "Mixed\nmopane",
#     "Exposed\nsoils"
# ]

# # 计算最大标签的宽度和高度
# max_label_width = max([max([len(word) for word in label.split("\n")]) for label in labels]) * 0.09
# max_label_height = len(max(labels, key=lambda s: len(s.split("\n"))).split("\n")) * 0.16

# # 根据标签尺寸设置画布大小
# num_columns = 2
# num_rows = num_classes // num_columns + num_classes % num_columns
# fig_width = num_columns * max_label_width
# fig_height = num_rows * max_label_height
# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # 计算方块位置
# for i in range(num_classes):
#     x = (i % num_columns) * max_label_width + max_label_width / 2
#     y = fig_height - (i // num_columns) * max_label_height - max_label_height / 2
#     ax.add_patch(mpatches.Rectangle((x - max_label_width / 2, y - max_label_height / 2), 
#                                     max_label_width, max_label_height, color=colors[i]))
#     ax.text(x, y, labels[i], ha='center', va='center', color='black', fontsize=10)  # 字体大小与"label_pu_vertical_adapted.png"保持一致

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, fig_height)

# plt.tight_layout(pad=0)
# plt.show()
# plt.savefig('label_bot_vertical_adapted.png', dpi=300)


### houston 2013 dataset

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置颜色数
# num_classes = 15

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     'Healthy\ngrass', 'Stressed\ngrass', 'Tree', 'Synthetic\ngrass', 'Soil', 
#     'Water', 'Residential', 'Commercial', 'Road', 'Highway', 
#     'Railway', 'Parking\nlot 1', 'Parking\nlot 2', 'Tennis\ncourt', 'Running\ntrack'
# ]

# # 创建一个临时的figure和ax来计算文本尺寸
# tmp_fig, tmp_ax = plt.subplots()
# max_text_width = 0
# max_text_height = 0
# for label in labels:
#     text = tmp_ax.text(0, 0, label, fontsize=10, ha='center', va='center', multialignment='center')
#     tmp_fig.canvas.draw()
#     bbox = text.get_window_extent()
#     max_text_width = max(max_text_width, bbox.width)
#     max_text_height = max(max_text_height, bbox.height)
# plt.close(tmp_fig)

# # 调整文本尺寸以适应所有文本
# max_label_width = max_text_width / plt.gcf().dpi
# max_label_height = max_text_height / plt.gcf().dpi

# # 增加额外空间以确保文本不超出方框
# extra_space = 0.1  # 增加一些额外的空间
# max_label_height += extra_space

# # 根据标签尺寸设置画布大小
# num_rows = 2
# num_columns = num_classes // num_rows + num_classes % num_rows
# fig_width = num_columns * max_label_width
# fig_height = num_rows * max_label_height
# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # 计算方块位置
# for i in range(num_classes):
#     x = (i // num_rows) * max_label_width + max_label_width / 2
#     y = fig_height - (i % num_rows) * max_label_height - max_label_height / 2
#     ax.add_patch(mpatches.Rectangle((x - max_label_width / 2, y - max_label_height / 2), 
#                                     max_label_width, max_label_height, color=colors[i]))
#     ax.text(x, y, labels[i], ha='center', va='center', color='black', fontsize=10)

# # 隐藏坐标轴和背景
# ax.axis('off')
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, fig_height)

# plt.tight_layout(pad=0)
# plt.show()
# plt.savefig('label_hu_vertical_adapted.png', dpi=300)





# ### botswana dataset 20250427

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # 设置类别数
# num_classes = 14

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Water", 
#     "Hippo\ngrass",
#     "Floodplain\ngrasses 1", 
#     "Floodplain\ngrasses 2",
#     "Reeds", 
#     "Riparian", 
#     "Firescar", 
#     "Island\ninterior",
#     "Acacia\nwoodlands", 
#     "Acacia\nshrublands",
#     "Acacia\ngrasslands", 
#     "Short\nmopane", 
#     "Mixed\nmopane",
#     "Exposed\nsoils"
# ]

# # 每行列数
# n_cols = 7
# n_rows = 2

# # 字体大小
# fontsize = 10

# # 临时画布测量文字宽度
# fig_tmp, ax_tmp = plt.subplots()
# text_widths = []
# text_heights = []
# for label in labels:
#     text = ax_tmp.text(0, 0, label, fontsize=fontsize, va='center', ha='center')
#     fig_tmp.canvas.draw()
#     bbox = text.get_window_extent()
#     width_inch = bbox.width / fig_tmp.dpi
#     height_inch = bbox.height / fig_tmp.dpi
#     text_widths.append(width_inch)
#     text_heights.append(height_inch)
# plt.close(fig_tmp)

# # 找最大宽度和最大高度
# max_text_width = max(text_widths)
# max_text_height = max(text_heights)

# # 设置框格宽高，稍微多一点padding
# block_width = max_text_width * 1.2
# block_height = max_text_height * 1.5  # 更紧凑（前面一般是2.0倍，这里压到1.8倍）

# # 画布大小
# fig_width = block_width * n_cols
# fig_height = block_height * n_rows

# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # 绘制小方块和文字
# for idx, (label, color) in enumerate(zip(labels, colors)):
#     col_idx = idx % n_cols
#     row_idx = idx // n_cols
#     x = col_idx * block_width
#     y = (n_rows - 1 - row_idx) * block_height
#     ax.add_patch(mpatches.Rectangle((x, y), block_width, block_height, color=color))
#     ax.text(x + block_width/2, y + block_height/2, label, ha='center', va='center', color='white', fontsize=fontsize)

# # 设置坐标轴
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, fig_height)
# ax.axis('off')

# plt.tight_layout(pad=0)
# plt.savefig('label_bot_real_fit_tight.png', dpi=300)
# plt.show()




# ############# Houston 2013 with one column
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # Houston 2013 标签
# num_classes = 15
# colors = sns.color_palette("hls", num_classes)
# labels = [
#     'Healthy\ngrass', 'Stressed\ngrass', 'Tree', 'Synthetic\ngrass', 'Soil', 'Water',
#     'Residential', 'Commercial', 'Road', 'Highway', 'Railway',
#     'Parking\nlot 1', 'Parking\nlot 2', 'Tennis\ncourt', 'Running\ntrack'
# ]

# # 每行列数
# n_cols = 8  # 第一行放8个
# n_rows = 2  # 第二行放7个（自动分配）

# # 字体大小
# fontsize = 10

# # 临时画布测量文字真实尺寸
# fig_tmp, ax_tmp = plt.subplots()
# text_widths = []
# text_heights = []
# for label in labels:
#     text = ax_tmp.text(0, 0, label, fontsize=fontsize, va='center', ha='center')
#     fig_tmp.canvas.draw()
#     bbox = text.get_window_extent()
#     width_inch = bbox.width / fig_tmp.dpi
#     height_inch = bbox.height / fig_tmp.dpi
#     text_widths.append(width_inch)
#     text_heights.append(height_inch)
# plt.close(fig_tmp)

# # 找最大宽度和最大高度
# max_text_width = max(text_widths)
# max_text_height = max(text_heights)

# # 框格大小，稍微放宽一点padding
# block_width = max_text_width * 1.2
# block_height = max_text_height * 1.3

# # 画布大小
# fig_width = block_width * n_cols
# fig_height = block_height * n_rows
# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # 绘制方块和文字
# for idx, (label, color) in enumerate(zip(labels, colors)):
#     col_idx = idx % n_cols
#     row_idx = idx // n_cols
#     x = col_idx * block_width
#     y = (n_rows - 1 - row_idx) * block_height  # 从上到下排
#     ax.add_patch(mpatches.Rectangle((x, y), block_width, block_height, color=color))
#     ax.text(x + block_width/2, y + block_height/2, label, ha='center', va='center', color='white', fontsize=fontsize)

# # 设置范围
# ax.set_xlim(0, fig_width)
# ax.set_ylim(0, fig_height)

# # 隐藏坐标轴
# ax.axis('off')

# plt.tight_layout(pad=0)
# plt.savefig('label_houston2013_2rows.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()



# #pavia university dataset20250427

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # Pavia University dataset
# num_classes = 9

# # 使用seaborn生成颜色
# colors = sns.color_palette("hls", num_classes)

# labels = [
#     "Asphalt",
#     "Meadows",
#     "Gravel",
#     "Trees",
#     "Painted\nmetal sheets",
#     "Bare soil",
#     "Bitumen",
#     "Self-blocking\nbricks",
#     "Shadows"
# ]

# # 字体大小
# fontsize = 10

# # 临时画布测量真实文字尺寸
# fig_tmp, ax_tmp = plt.subplots()
# text_widths = []
# text_heights = []
# for label in labels:
#     text = ax_tmp.text(0, 0, label, fontsize=fontsize, va='center', ha='center')
#     fig_tmp.canvas.draw()
#     bbox = text.get_window_extent()
#     width_inch = bbox.width / fig_tmp.dpi
#     height_inch = bbox.height / fig_tmp.dpi
#     text_widths.append(width_inch)
#     text_heights.append(height_inch)
# plt.close(fig_tmp)

# # 最大文字宽度和高度
# max_text_width = max(text_widths)
# max_text_height = max(text_heights)

# # 框格尺寸，稍微加点padding
# block_width = max_text_width * 1.2
# block_height = max_text_height * 1.3

# # 设定每行列数
# first_row_num = 5
# second_row_num = 4

# max_cols = max(first_row_num, second_row_num)

# # 画布大小
# fig_width = block_width * max_cols
# fig_height = block_height * 2
# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # 绘制方块和文字
# for idx, (label, color) in enumerate(zip(labels, colors)):
#     if idx < first_row_num:
#         row_idx = 0
#         col_idx = idx
#     else:
#         row_idx = 1
#         col_idx = idx - first_row_num
#     x = col_idx * block_width
#     y = (1 - row_idx) * block_height
#     ax.add_patch(mpatches.Rectangle((x, y), block_width, block_height, color=color))
#     ax.text(x + block_width/2, y + block_height/2, label, ha='center', va='center', color='white', fontsize=fontsize)

# # 设置范围
# ax.set_xlim(0, block_width * max_cols)
# ax.set_ylim(0, block_height * 2)
# ax.axis('off')

# plt.tight_layout(pad=0)
# plt.savefig('label_pu_2rows.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
