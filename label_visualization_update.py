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
#     print(cmap)
    
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
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI classification evaluation")
    parser.add_argument("--model", type=str, default='ssftt')
    parser.add_argument("--dataset_name", type=str, default="hu")
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

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes + 1)):    # hls changed to Paired
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    # # load model and weights
    # model = get_model(opts.model, opts.dataset_name, opts.patch_size, opts.trans_type)
    # print('loading weights from %s' % opts.weights + '/model_best.pth')
    # model = model.to(device)
    # model.load_state_dict(torch.load(os.path.join(opts.weights, 'model_best.pth')))
    # model.eval()

    # # testing model: metric for the whole HSI, including train, val, and test
    # probabilities = test(model, opts.weights, image, opts.patch_size, num_classes, device=device)
    # prediction = np.argmax(probabilities, axis=-1)

    # run_results = metrics(prediction, gt, n_classes=num_classes)

    # prediction[gt < 0] = -1   # mask the no label points

    # color results
    colored_gt = color_results(gt+1, palette)
    # colored_pred = color_results(prediction+1, palette)



    # # Create a colormap with 'classes' number of random colors
    # # cmap = plt.cm.get_cmap('tab20', classes)
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(colored_pred)
    
    # # Create a colorbar with labels
    # cbar = plt.colorbar(im, ax=ax, ticks=np.arange(classes))
    # cbar.set_label('Classes', rotation=270, labelpad=15)
    # cbar.set_ticklabels(['Class {}'.format(i+1) for i in range(classes)])





    import numpy as np
    import matplotlib.pyplot as plt
    import imageio

    # 读取原始图像
    image = imageio.imread('false_color_image_hu.png')

    # 原始图像的尺寸
    image_width, image_height = image.shape[1], image.shape[0]

    # 计算以英寸为单位的图像尺寸
    fig_width = image_width / 100.0
    fig_height = image_height / 100.0

    # 使用计算出的英寸值创建图像
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 显示图像
    ax.imshow(image)
    ax.axis('off')
    # ... 这里可以添加局部放大等其他代码 ...

    # 保存图像
    plt.savefig('output_image_path.png', dpi=100, bbox_inches='tight', pad_inches=0)

    plt.close()



