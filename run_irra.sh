#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name DCEL \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'id+unc+sdm+sdm_loc+mlm' \
--num_epoch 70