export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch train_segformer.py