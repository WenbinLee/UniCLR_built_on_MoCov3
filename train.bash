#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      MoCo-V3         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'



python main_moco.py \
  -a vit_small -b 512 \
  --epochs=200  \
  --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  /data/ssl/ImageNet2012

