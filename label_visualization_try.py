# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import ListedColormap

# # Generate a mock 2D array with random values between 0 and 14 (inclusive)
# data = np.random.randint(15, size=(100, 100))

# num_classes = 15
# colors = sns.color_palette("hls", num_classes)
# cmap = ListedColormap(colors)

# fig, ax = plt.subplots(figsize=(8, 8))
# pixel_plot = ax.imshow(data, cmap=cmap)

# # Create a new axes for the segmented colorbar on the right side of the main plot
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

# # Plot each rectangle for the colorbar
# height = 1 / num_classes
# for idx, color in enumerate(colors):
#     cbar_ax.add_patch(plt.Rectangle((0, idx * height), 1, height - 0.01, fc=color))
#     cbar_ax.text(1.5, idx * height + height/2, str(idx), va='center', ha='left')

# cbar_ax.set_xlim(0, 1)
# cbar_ax.set_ylim(0, 1)
# cbar_ax.axis('off')  # Turn off the axis of the colorbar
# filename = 'false_color_image_hu_colorbar.png'
# plt.savefig(filename, dpi=300)
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Create a random 2D array
data = np.random.random((100, 100))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(data, cmap="viridis")

# Define the bounding box of the region to be zoomed in on
x1, x2, y1, y2 = 40, 60, 40, 60

# Create inset of required size and position
axins = inset_axes(ax, 
                   width='30%',  # width = 30% of parent_bbox width
                   height='30%', # height : 30% of parent_bbox height
                   loc='upper right')

# Display the same data on both axes
axins.imshow(data, cmap="viridis")

# Apply the x-limits and y-limits to the inset axes
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

filename = 'false_color_image_hu_zoom.png'
plt.savefig(filename, dpi=300)
plt.show()