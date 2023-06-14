#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/compat 
cd /Mediapipe-Extract-Videos

python3 create_batches.py --num_batches ${SLURM_ARRAY_TASK_COUNT} --use_hands

python3 extract_mediapipe.py \
--jobArrayNum ${SLURM_ARRAY_TASK_ID} \
--inputDirectory /data/sign_language_videos/ \
--outputDirectory /Extracted_Hands \
--useHands
