#!/bin/bash
source ~/.bashrc
conda activate TFG-DM-lagr

export OPENAI_LOGDIR="/data_8T/Conghui/lagr/data-train"

DATA_FLAGS="--dataset_path /data_8T/Conghui/lagr/lagr-data/split/Lagr_u3c_diffusion_1024_splits.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 128,64 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--use_continuous_diffusion True --diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"

mpiexec -n 4 python ../../scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
