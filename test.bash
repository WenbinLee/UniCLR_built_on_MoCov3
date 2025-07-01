#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      MoCo-V3 / UniCLR Testing        +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'


python main_lincls.py \
  -a vit_small -b 256 --lr=30 \
  --dist-url 'tcp://localhost:10010' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained ./results2/simWhitening_300epoch_Batchsize_1024_Lr_0.0001/vit_small_300Epoch_Batch_1024_adamw_0.001/checkpoint_0299.pth.tar \
  /data/ssl/ImageNet2012