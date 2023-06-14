#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/compat 
cd /New-Decoding-Technique

python3 create_batches.py --num_batches ${SLURM_ARRAY_TASK_COUNT}

python3 decode_split_by_length.py \
--job_array_num ${SLURM_ARRAY_TASK_ID} \
--ffmpeg_loglevel info \
--backup_dir /data/sign_language_videos/raw_staging \
--dest_dir /data/sign_language_videos/split \
--video_dim 1944 2592 \
--num_threads 24
