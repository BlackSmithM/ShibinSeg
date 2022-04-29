import os
import numpy as np
from tqdm import tqdm


def split_sar_optic(root):
    save_image = os.path.join(root, 'label')
    image_list = os.listdir(save_image)

    np.random.shuffle(image_list)

    valid_list = image_list[::5]
    train_list = [x for x in image_list if x not in valid_list]


    with open(os.path.join(root, 'train_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(train_list):
            f.write(root + '/sar/' + os.path.splitext(each)[0] + '.png' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')

    with open(os.path.join(root, 'val_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(valid_list):
            f.write(root + '/sar/' + os.path.splitext(each)[0] + '.png' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')


if __name__ == '__main__':
    root = '/workspace/MaShibin/ShibinSeg/GF3_data/'
    split_sar_optic(root)
    # split_optic(root)
