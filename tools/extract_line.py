import cv2
import numpy as np
from PIL import Image
from skimage import morphology


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
    num_classes = 15  # num_classes <= len(palette)

    colors = get_color_map_list(num_classes)
    r = Image.fromarray(label.astype('uint8'), mode='P')
    r.putpalette(colors)
    return r


label = Image.open('4.png')
label = np.array(label)
label[label != 0] = 1

skeleton_class4 = label
kernel_skeleton_class4_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
skeleton_class4 = cv2.morphologyEx(skeleton_class4, cv2.MORPH_CLOSE, kernel_skeleton_class4_1, iterations=1)
skeleton_class4 = morphology.skeletonize(skeleton_class4)
# kernel_skeleton_class4_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
# skeleton_class4 = cv2.dilate(skeleton_class4.astype(np.uint8), kernel_skeleton_class4_2, iterations=                                           1)
# kernel_skeleton_class4_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
# skeleton_class4 = cv2.erode(skeleton_class4.astype(np.uint8), kernel_skeleton_class4_3, iterations=1)

skeleton_class4 = create_pseudo_color_image(skeleton_class4)
skeleton_class4.save('output/line.png')
print('Done!')


"""
武汉大学使用的方法
"""
# selem = morphology.disk(2)
# label = morphology.binary_dilation(label, selem)
# label = morphology.thin(label)
# label = create_pseudo_color_image(label)
# label.save('output/line.png')


