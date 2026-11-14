#!/bin/bash
source ~/.bashrc
conda activate TFG-DM-lagr

export OPENAI_LOGDIR="/data_8T/Conghui/lagr/data-sample/level_5/0.01"

DATA_FLAGS="--task scale_reconstruction_level5 --dataset_path /data_8T/Conghui/lagr/lagr-data/split/Lagr_u3c_diffusion_1024_splits.h5 --dataset_name valid"
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 128,64 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
SAMPLE_FLAGS="--num_samples 65536 --batch_size 64 --model_path /home/Conghui/TFG-DM-lagr/datasets/lagr/dataset/dps-sample/ema_0.9999_400000.pt --seed 42"
GUIDANCE_FLAGS="--guidance_name dps --guidance_strength 0.01"
nohup python /home/Conghui/TFG-DM-lagr/scripts/tfg_sample_wave_sum_1024.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $GUIDANCE_FLAGS > 0.01.out 2>&1 & 