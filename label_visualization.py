# import numpy as np
# import matplotlib.pyplot as plt

# def generate_sample_labels(size=(100, 100), classes=5):
#     """Generate a sample label array with given size and number of classes."""
#     return np.random.randint(0, classes, size)

# def visualize_labels(labels):
#     """Visualize the labels with a color map."""
#     # Number of unique classes in the labels
#     classes = len(np.unique(labels))
    
#     # Create a colormap with 'classes' number of random colors
#     cmap = plt.cm.get_cmap('tab20', classes)
    
#     fig, ax = plt.subplots()
#     im = ax.imshow(labels, cmap=cmap)
    
#     # Create a colorbar with labels
#     cbar = plt.colorbar(im, ax=ax, ticks=np.arange(classes))
#     cbar.set_label('Classes', rotation=270, labelpad=15)
#     cbar.set_ticklabels(['Class {}'.format(i+1) for i in range(classes)])
#     plt.savefig("output_image_label.png")
#     plt.show()

# # Generate and visualize sample labels
# sample_labels = generate_sample_labels()
# visualize_labels(sample_labels)




import os
import torch
import argparse
import seaborn as sns
import numpy as np
from utils.dataset import load_mat_hsi
from models.get_model import get_model
from train import test
from utils.utils import metrics, show_results
import imageio
import matplotlib.pyplot as plt
from skimage.io import imsave
from scipy import io



def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d


def linear_stretch_per_band(band, lower_percent=2, higher_percent=98):
    """Linear stretch"""
    lower_bound = np.percentile(band, lower_percent)
    upper_bound = np.percentile(band, higher_percent)
    stretched_band = np.clip(band, lower_bound, upper_bound)
    return (stretched_band - lower_bound) / (upper_bound - lower_bound)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI classification evaluation")
    parser.add_argument("--model", type=str, default='ssftt')
    parser.add_argument("--dataset_name", type=str, default="pu")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--weights", type=str, default="./checkpoints/ssftt/hu/0")
    parser.add_argument("--outputs", type=str, default="./results")
    parser.add_argument("--trans_type", type=int, default=0)


    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    print("dataset: {}".format(opts.dataset_name))
    print("patch size: {}".format(opts.patch_size))
    print("model: {}".format(opts.model))


    # # houston 2013 dataset
    # image = io.loadmat(os.path.join(opts.dataset_dir, opts.dataset_name, "HU_cube.mat"))
    # image = image['HU_cube']

    # # Assuming the data is in [height, width, bands]
    # # houston 2013 dataset
    # red_band = image[:, :, 60]   # 50th band as red
    # green_band = image[:, :, 40] # 30th band as green
    # blue_band = image[:, :, 15]  # 10th band as blue

    # # bot dataset
    # # image = io.loadmat(os.path.join(opts.dataset_dir, opts.dataset_name, "Botswana.mat"))
    # # image = image['Botswana']
    # # red_band = image[:, :, 60]   # 50th band as red
    # # green_band = image[:, :, 30] # 30th band as green
    # # blue_band = image[:, :, 20]  # 10th band as blue

    # # pu dataset
    # # image = io.loadmat(os.path.join(opts.dataset_dir, opts.dataset_name, "PaviaU.mat"))
    # # image = image['paviaU']
    # # red_band = image[:, :, 60]   # 50th band as red
    # # green_band = image[:, :, 40] # 30th band as green
    # # blue_band = image[:, :, 15]  # 10th band as blue

    # # Stack bands along the third dimension to create an RGB image
    # false_color_image = np.stack((red_band, green_band, blue_band), axis=2)

    # # Normalize the image to [0,1]
    # false_color_image = (false_color_image - np.min(false_color_image)) / (np.max(false_color_image) - np.min(false_color_image))
    # imsave('false_color_image_hu.png', (false_color_image * 255).astype(np.uint8))




# #### hu dataset#####
# # 加载图像
# image = io.loadmat(os.path.join(opts.dataset_dir, opts.dataset_name, "HU_cube.mat"))
# image = image['HU_cube']

# # 选择波段
# red_band = image[:, :, 60]   # 60th band as red
# green_band = image[:, :, 40] # 40th band as green
# blue_band = image[:, :, 15]  # 15th band as blue

# # 对每个波段进行线性拉伸
# red_stretched = linear_stretch_per_band(red_band)
# green_stretched = linear_stretch_per_band(green_band)
# blue_stretched = linear_stretch_per_band(blue_band)

# # 组合拉伸后的波段
# false_color_image = np.stack((red_stretched, green_stretched, blue_stretched), axis=2)

# # 将图像转换为8位并保存
# false_color_image_uint8 = (false_color_image * 255).astype(np.uint8)
# imsave('false_color_image_hu_stretched.png', false_color_image_uint8)




#### pu dataset#####
# 加载图像
image = io.loadmat(os.path.join(opts.dataset_dir, opts.dataset_name, "PaviaU.mat"))
image = image['paviaU']

# 选择波段
red_band = image[:, :, 40]   # 60th band as red
green_band = image[:, :, 30] # 40th band as green
blue_band = image[:, :, 20]  # 15th band as blue

# 对每个波段进行线性拉伸
red_stretched = linear_stretch_per_band(red_band)
green_stretched = linear_stretch_per_band(green_band)
blue_stretched = linear_stretch_per_band(blue_band)

# 组合拉伸后的波段
false_color_image = np.stack((red_stretched, green_stretched, blue_stretched), axis=2)

# 将图像转换为8位并保存
false_color_image_uint8 = (false_color_image * 255).astype(np.uint8)
imsave('false_color_image_pu_stretched.png', false_color_image_uint8)
