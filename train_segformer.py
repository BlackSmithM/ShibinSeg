from paddleseg.models import *
from paddleseg.datasets import Dataset
import paddle
from paddleseg.core import train
from paddleseg.models.losses import *
import paddleseg.transforms as T
from clr import CLR
import paddle.distributed as dist


nranks = paddle.distributed.ParallelEnv().nranks
if nranks > 1:
    dist.init_parallel_env()

batch_size = 8
epoch = 200
base_lr = 0.005
model = SegFormer_B5(num_classes=5,
                     pretrained='/workspace/MaShibin/ShibinSeg/pretrained/mix_vision_transformer_b5.pdparams')
paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model_info = paddle.summary(model, (1, 3, 256, 256))
# print(model_info)

transforms = [
    T.Normalize(),
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.ResizeStepScaling(),
    T.Resize((256, 256))
]
transforms_val = [
    T.Normalize(),
    T.Resize((256, 256))
]

train_dataset = Dataset(
    dataset_root='',
    transforms=transforms,
    train_path=r'/workspace/MaShibin/ShibinSeg/GF3_data/train_list.txt',
    num_classes=5,
    mode='train',
    ignore_index=255
)
val_dataset = Dataset(
    dataset_root='',
    transforms=transforms_val,
    val_path=r'/workspace/MaShibin/ShibinSeg/GF3_data/val_list.txt',
    num_classes=5,
    mode='val',
    ignore_index=255
)

total_step = epoch * len(train_dataset) // batch_size // nranks
print('total step is {}'.format(total_step))
# lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.001, power=0.9, decay_steps=total_step, end_lr=0)
# lr = CLR(train_set=train_dataset, learning_rate=0.001, max_lr=0.006, epochs_per_cycle=4,
#          batch_size=batch_size)
lr = paddle.optimizer.lr.LinearWarmup(learning_rate=base_lr, warmup_steps=total_step // epoch * 2,
                                      start_lr=base_lr / 10,
                                      end_lr=base_lr)
# optimizer = paddle.optimizer.SGD(lr, parameters=model.parameters())
optimizer = paddle.optimizer.Lamb(lr, parameters=model.parameters())

losses = {}
losses['types'] = [MixedLoss([CrossEntropyLoss(), LovaszSoftmaxLoss()], [0.8, 0.2])]
losses['coef'] = [1]
# losses['types'] = [CrossEntropyLoss()]
# losses['coef'] = [1]
# losses['types'] = [SemanticConnectivityLoss(ignore_index=255, max_pred_num_conn=10, use_argmax=True)]
# losses['coef'] = [1]

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='saved_model/GF3',
    iters=total_step,
    batch_size=batch_size,
    save_interval=total_step // epoch * 2,
    log_iters=total_step // epoch // 2,
    losses=losses,
    use_vdl=False,
    num_workers=16,
    keep_checkpoint_max=2,
    resume_model=None)
