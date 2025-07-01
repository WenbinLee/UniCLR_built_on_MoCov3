#!/bin/bash
# echo '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
# echo '+++++   MoCo-V3 Alignment and Uniform measure Testing        +++++'
# echo '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


python main_moco_align_uniform.py \
  -a vit_small -b 256 --lr=30 \
  --dist-url 'tcp://localhost:10011' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained ./results/MoCo_300epoch_Batchsize1024_Lr_0.001/vit_small_300Epoch_Batch_1024_adamw_0.001/checkpoint_0299.pth.tar \
  /data/ssl/ImageNet2012


