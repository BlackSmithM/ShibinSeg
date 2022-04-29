# ShibinSeg
Segmentation based on paddleseg2.5
## 环境要求
已在CUDA11.1,CUDA10.2  WIN,UBUNTU上测试

- PaddlePaddle >= 2.0
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1 
- cuDNN >= 7.6

使用高阶API进行训练和推理
'''
训练启动命令 bash dist_train.sh
推理启动命令 python Infer_Big.py
'''
优化了推理部分的代码，使用多batch size加速推理过程

修改了dataset部分代码，使用完整路径进行数据索引

数据格式参考[data](./data)目录
