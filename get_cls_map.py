import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)



    # ### add the legend ###
    # # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    # labels = {
    #         0: "Undefined", 
    #         1: "Healthy grass",
    #         2: "Stressed grass",
    #         3: "Synthetic grass",
    #         4: "Trees",
    #         5: "Soil",
    #         6: "Water",
    #         7: "Residential",
    #         8: "Commercial",
    #         9: "Road",
    #         10: "Highway",
    #         11: "Railway",
    #         12: "Parking Lot 1",
    #         13: "Parking Lot 2",
    #         14: "Tennis Court",
    #         15:"Running Track"}  
    # cmap = {0: np.array([0, 0, 0]) / 255.,
    #         1: np.array([147, 67, 46]) / 255.,
    #         2: 
    #         3:
    #         4:
    #         5:
    #         6:
    #         7:
    #         8:
    #         9:
    #         10:
    #         11:
    # }

    # patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    # ax.legend(handles=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # ### add the legend ###

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0



def get_cls_map(y):

    gt = y.flatten()

    y_gt = list_to_colormap(gt)

    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    classification_map(gt_re, y, 300,
                       'classification_maps/' + 'gt.png')
    print('------Get classification maps successful-------')





if __name__ == "__main__":

    labels = sio.loadmat('./datasets/hu/gt.mat')['gt']

    # temp_arr_0 = np.zeros((labels.shape)).astype(int)

    # #### commercial = 8, Highway = 10,  Parking lot 1 = 12
    # temp_arr_0[labels == 8] = 8    

    # get_cls_map(temp_arr_0)

    get_cls_map(labels)