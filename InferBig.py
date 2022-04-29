from paddleseg.models import *
import paddleseg.transforms as T
import os
from paddleseg.core import predict
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np

model = SegFormer_B5(num_classes=5)
# model = SegmentationTransformer(backbone=ViT_large_patch16_384(),num_classes=5)
# model = SFNet(backbone=ResNet50_vd(),num_classes=5,backbone_indices=[0, 1, 2, 3])
# model = DNLNet(backbone=ResNet101_vd(),num_classes=5)

model_name = 'segformer'
model_path = 'output/segformer_korea_cloud256/best_model/model.pdparams'

transforms = T.Compose([
    T.Normalize()
])


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


# image_path = 'data/our_data' # 也可以输入一个包含图像的目录
image_path = '/workspace/MaShibin/DATA/korea/cloud.jpg'
image_list, image_dir = get_image_list(image_path)

paddle.flops(model, input_size=[1, 3, 256, 256])
# print(slim.analysis.flops(model, inputs=image))
predict(
    model,
    model_path=model_path,
    transforms=transforms,
    image_list=image_list,
    image_dir=image_dir,
    save_dir='vis_results/korea_cloud',
    save_name=model_name,
    is_slide=True,
    stride=(256, 256),
    crop_size=(256, 256),
    batch_size=40,
    aug_pred=True,

)

# 构建验证用的transforms
# transforms = T.Compose([
#     T.Normalize(512,512),
#     T.Normalize()
# ])

# image_path = 'data/crop/image' # 也可以输入一个包含图像的目录
# image_list, image_dir = get_image_list(image_path)
# predict(
#         model,
#         model_path='output/iter_14500/model.pdparams',
#         transforms=transforms,
#         image_list=image_list,
#         image_dir=image_dir,
#         save_dir='results/crop'
#     )
