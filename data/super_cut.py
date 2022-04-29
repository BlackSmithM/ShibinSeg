# 不用gdal，裁剪：


import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm, trange
import time
from PIL import Image


def get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    # color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = color_map[3:]
    return color_map

def create_pseudo_color_image(label):
    # label = Image.fromarray(label)
    num_classes = 20  # num_classes <= len(palette)

    colors = get_color_map_list(num_classes)
    r = Image.fromarray(label.astype('uint8'), mode='P')
    r.putpalette(colors)
    return r

def crop(cnt, data_dir, img_file, label_dir, label_file, save_file_dir, save_label_dir):
    # 加载数据
    file_path = os.path.join(data_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    dataset = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # label = cv2.imread(label_path, cv2.IMREAD_COLOR)
    label = Image.open(label_path)
    label = np.array(label)
    label -= 1
    rows, cols, channel = dataset.shape

    step_rows_count = rows // step + 1
    step_cols_count = cols // step + 1

    new_rows = step * (step_rows_count + 1)
    new_cols = step * (step_cols_count + 1)

    new_dataset = np.full([new_rows, new_cols, channel], 255, dtype="int")
    new_label = np.full([new_rows, new_cols], 255, dtype="int")

    new_dataset[step // 2:step // 2 + rows, step // 2:step // 2 + cols, :] = dataset
    new_label[step // 2:step // 2 + rows, step // 2:step // 2 + cols] = label

    # del dataset
    # del label
    t = trange(step_rows_count)
    small_cnt = 0
    for i in t:

        for j in range(step_cols_count):
            # print("cut: img {} of {}, row {} of {}, col {} of {}".format(
            #     cnt, len(label_files), i, step_rows_count, j, step_cols_count),
            #     end="\r"
            # )
            t.set_postfix({'cnt': small_cnt, 'row': i, 'col': j})
            xoffset = j * step
            yoffset = i * step
            width = split
            height = split

            data_cuted = new_dataset[yoffset: yoffset + height, xoffset:xoffset + width]
            label_cuted = new_label[yoffset: yoffset + height, xoffset:xoffset + width]

            # file_save_path = os.path.join(save_file_dir, "{}.jpg".format(cnt_small))
            file_save_path = os.path.join(save_file_dir, img_file[:-4]+"{}.png".format(small_cnt))
            label_save_path = os.path.join(save_label_dir, img_file[:-4]+"{}.png".format(small_cnt))
            cv2.imwrite(file_save_path, data_cuted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(label_save_path, label_cuted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            label_cuted = create_pseudo_color_image(label_cuted)
            label_cuted.save(label_save_path)
            # print("file_save_path: ", file_save_path, end = "\r")
            
            small_cnt += 1
    
    # 保存原标签的长宽
    # label_info_save_path = os.path.join(save_label_dir, "info.txt")
    # info = np.array([rows, cols], dtype="int")
    # np.savetxt(label_info_save_path, info, fmt="%i")
    



start=time.time()


path = r'/workspace/MaShibin/ATianZhi/DATA/GID_segdata/'

# 要拆分的图片大小
split = 512
# 歩长
step = 256

# 图片所在路径
data_dir = os.path.join(path, "rgb")
label_dir = os.path.join(path, 'label')
files = os.listdir(data_dir)
label_files = os.listdir(label_dir)

# 图片保存路径
save_path = os.path.join(path, "crop")
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 训练时放到同一文件夹即可
save_file_dir = os.path.join(save_path, "image")
save_label_dir = os.path.join(save_path, "label")
if not os.path.exists(save_file_dir):
    os.mkdir(save_file_dir)
if not os.path.exists(save_label_dir):
    os.mkdir(save_label_dir)

cnt = 0
P = Pool(10)
for img_file, label_file in zip(files, label_files):
    P.apply_async(crop, (cnt, data_dir, img_file, label_dir, label_file, save_file_dir, save_label_dir))
    # crop(cnt, data_dir, img_file, label_dir, label_file, save_file_dir, save_label_dir)
P.close()
P.join()
end=time.time()
print('Running time: %s Seconds'%(end-start))
print("\ndone.")