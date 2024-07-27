import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import pandas as pd
import torch
import pandas
import cv2
import numpy as np
import yaml
import archs
from skimage import morphology



def highlight_spot_func(x):
    # print(x)
    # cv2.imshow('froth', x)
    retval, x1 = cv2.threshold(x, 99, 255, cv2.THRESH_BINARY)
    # cv2.imshow('x1', x1)
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    x2 = cv2.morphologyEx(x1, cv2.MORPH_OPEN, kernel_3)
    # cv2.imshow('x2', x2)
    x3 = cv2.morphologyEx(x2, cv2.MORPH_CLOSE, kernel_3)
    # cv2.imshow('x3', x3)
    area_sum = len(x3[x3 == 255])
    # print(area_sum)
    contours, hierarchy = cv2.findContours(x3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    weight_map = np.full(x.shape, 0.0005)
    i = 0
    for c in contours:
        painted_image = np.zeros(x.shape)
        painted_image = cv2.drawContours(painted_image, contours, i, 255, -1)
        i += 1
        area = cv2.contourArea(c)
        # weight_number = np.exp((area / area_sum) - 1) ** 2.5
        weight_number = (area / area_sum) ** 0.5
        weight_map[painted_image == 255] = weight_number

    # cv2.imshow('highlight_spot', weight_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return weight_map


def depth_func(x):
    # print(x)
    max = np.max(x)
    min = np.min(x)
    normal_x = (x - min) / (max - min)
    normal_x = np.exp(normal_x - 1)

    # normal_x_show = normal_x * 255
    # cv2.imshow('depth_map', normal_x_show.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return normal_x


def BSD_weight_func(x):
    ret, x_bin = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
    x_bin = 255 - x_bin
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(x_bin.astype(np.uint8),
                                                                            connectivity=4, ltype=None)
    result = np.zeros((x.shape[0], x.shape[1]))  
    mean_bubble = x.size / num_labels
    for i in range(1, num_labels):
        mask = labels == i 
        weight_number = stats[i][4] / mean_bubble
        weight_number = 1 / (1 + np.exp(-(weight_number - 1)))
        result[mask] = weight_number
    result_add = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 128)
    weight_window = [0.125, 0.125, 0.125, 0.125, 0.000, 0.125, 0.125, 0.125, 0.125]
    for i in range(1, result_add.shape[0] - 1):
        for j in range(1, result_add.shape[1] - 1):
            value_window = []
            if result_add[i, j] == 0:
                for ii in range(i - 1, i + 2):
                    for jj in range(j - 1, j + 2):
                        value_window.append(result_add[ii, jj])
                result_add[i, j] = sum(np.multiply(value_window, weight_window))
    weight_map = result_add[1:251, 1:311]

    # cv2.imshow('BSD', (weight_map * 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return weight_map


def fusion_process(depth_map, spot_map_L, spot_map_R, BSD_map_L, BSD_map_R):
    # print(spot_map.shape, BSD_map.shape)
    x_ = spot_map_L * 0.25 + spot_map_R * 0.25 + BSD_map_L * 0.25 + BSD_map_R * 0.25
    fusion_FISM = medianBlur(x_, depth_map)
    fusion_FISM = 1 / (1 + np.exp(-1 * fusion_FISM))

    return fusion_FISM


def std_fusion_process(depth_map, spot_map_L, spot_map_R, BSD_map_L, BSD_map_R):
    depth_map_mean, depth_map_std = cv2.meanStdDev(depth_map)
    spot_mapL_mean, spot_mapL_std = cv2.meanStdDev(spot_map_L)
    spot_mapR_mean, spot_mapR_std = cv2.meanStdDev(spot_map_R)
    BSD_mapL_mean, BSD_mapL_std = cv2.meanStdDev(BSD_map_L)
    BSD_mapR_mean, BSD_mapR_std = cv2.meanStdDev(BSD_map_R)
    depth_map_vk = 1 / np.exp(depth_map_std)
    spot_mapL_vk = 1 / np.exp(spot_mapL_std)
    BSD_mapL_vk = 1 / np.exp(BSD_mapL_std)
    spot_mapR_vk = 1 / np.exp(spot_mapR_std)
    BSD_mapR_vk = 1 / np.exp(BSD_mapR_std)
    # vk = [depth_map_std, spot_map_std, BSD_map_std]
    # vk = np.exp(vk) / np.sum(np.exp(vk))
    # std_fusion = vk[0] * depth_map + vk[1] * spot_map + vk[2] * BSD_map
    std_fusion = depth_map_vk * depth_map + spot_mapL_vk * 1.5 * spot_map_L + 0.4 * BSD_mapL_vk * BSD_map_L + \
                 spot_mapR_vk * 1.5 * spot_map_R + 0.4 * BSD_mapR_vk * BSD_map_R

    return std_fusion


def medianBlur(disparity_map, weight_map, ksize=3):
    rows, cols = disparity_map.shape[:2]
    half = ksize // 2
    startSearchRow = half
    endSearchRow = rows - half - 1
    startSearchCol = half
    endSearchCol = cols - half - 1
    dst = np.zeros((rows, cols))
    for y in range(startSearchRow, endSearchRow):
        for x in range(startSearchCol, endSearchCol):
            dis_window = []
            weight_window = []
            for i in range(y - half, y + half + 1):
                for j in range(x - half, x + half + 1):
                    dis_window.append(disparity_map[i][j])
                    # print(weight_map[i][j])
                    weight_window.append(weight_map[i][j])
            weight_window = np.array(weight_window)
            weight_window = weight_window / sum(weight_window)
            dis_window = np.array(dis_window)
            weighted_sum = sum(np.multiply(dis_window, weight_window))
            # print(type(weighted_sum), weighted_sum)
            # window = np.sort(window, axis=None)
            # if len(window) % 2 == 1:
            #     medianValue = window[len(window) // 2]
            # else:
            #     medianValue = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
            dst[y][x] = weighted_sum
    return dst


def image_show_func(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    x = (np.exp(x) - 1) / (np.exp(1) - 1)
    # x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # x = (np.exp(2*(x - 0.5)) - np.exp(2*(-x + 0.5))) / (np.exp(2*(x - 0.5)) + np.exp(2*(-x + 0.5)))
    x = (x * 255).astype(np.uint8)

    return x


def write_FISM_func(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    x = (x * 255).astype(np.uint8)

    return x

if __name__ == '__main__':
    depthimage_path = 'depth'
    froth_image_path_L = '../calibration/cali_after_250/L'
    froth_image_path_R = '../calibration/cali_after_250/R'
    videos = os.listdir(depthimage_path)
    videos.sort()
    disparity = 80
    for video_item in videos:
        print(video_item)
        frames = os.listdir(os.path.join(depthimage_path, video_item))
        frames.sort()
        for frame_item in frames:
            print(frame_item)
            depth_image = pd.read_csv(os.path.join(depthimage_path, video_item, frame_item), header=None)
            depth_weight_map = depth_func(depth_image.to_numpy()[:, disparity:])

            frame_item_number = frame_item.split('.')[0]
            froth_image_L = cv2.imread(os.path.join(froth_image_path_L, video_item, frame_item_number + '.jpg'), 0)
            spot_weight_map_L = highlight_spot_func(froth_image_L[:, disparity:])
            froth_image_R = cv2.imread(os.path.join(froth_image_path_R, video_item, frame_item_number + '.jpg'), 0)
            spot_weight_map_R = highlight_spot_func(froth_image_R[:, :390 - disparity])

            BSD_froth_L = cv2.imread(os.path.join('seg_image\\L', video_item, frame_item_number + '.jpg'), 0)
            # BSD_froth = 255 - BSD_froth
            BSD_weight_map_L = BSD_weight_func(BSD_froth_L[:, disparity:])
            BSD_froth_R = cv2.imread(os.path.join('seg_image\\R', video_item, frame_item_number + '.jpg'), 0)
            BSD_weight_map_R = BSD_weight_func(BSD_froth_R[:, :390 - disparity])

            FISM_image = fusion_process(depth_weight_map, spot_weight_map_L, spot_weight_map_R, BSD_weight_map_L,
                                        BSD_weight_map_R)

            # write FISM
            FISM_write = write_FISM_func(FISM_image)
            if not os.path.exists(os.path.join('froth saliency map', video_item)):
                os.makedirs(os.path.join('froth saliency map', video_item))
            np.savetxt(os.path.join('froth saliency map', video_item, frame_item_number + '.csv'), FISM_write, delimiter=',')

            # mean_fusion_image = depth_weight_map * 1/5 + spot_weight_map_L * 1/5 + BSD_weight_map_L * 1/5 + \
            #                     spot_weight_map_R * 1/5 + BSD_weight_map_R * 1/5
            # std_fusion_image = std_fusion_process(depth_weight_map, spot_weight_map_L, spot_weight_map_R,
            #                                       BSD_weight_map_L, BSD_weight_map_R)
            #
            FISM_image_show = image_show_func(FISM_image)
            cv2.imshow('FISM', FISM_image_show)
            # cv2.imwrite('FISM com//FISM.jpg', FISM_image_show)
            # mean_fusion_image_show = image_show_func(mean_fusion_image)
            # cv2.imshow('mean_fusion', mean_fusion_image_show)
            # cv2.imwrite('FISM com//mean_fusion.jpg', mean_fusion_image_show)
            # std_fusion_image_show = image_show_func(std_fusion_image)
            # cv2.imshow('std_fusion', std_fusion_image_show)
            # cv2.imwrite('FISM com//std_fusion.jpg', std_fusion_image_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

