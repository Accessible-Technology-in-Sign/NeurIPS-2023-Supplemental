#!/bin/bash

#SBATCH --job-name=Mediapipe-Extract-Videos-Hands
#SBATCH --account=gts-ts133-paid
#SBATCH --nodes=1 --ntasks-per-node=24 --mem=0
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=Extraction-Hands%a.out
#SBATCH --mail-type=NONE
#SBATCH --array=1-100

cd $SLURM_SUBMIT_DIR

apptainer exec --nv \
--bind /storage/home/hcoda1/9/gdeshpande9/p-ts133-0/Mediapipe-Extract-Videos:/Mediapipe-Extract-Videos \
--bind ~/scratch/Videos:/data/sign_language_videos \
--bind ~/scratch/Extracted_Hands:/Extracted_Hands \
/storage/coda1/p-ts133/0/containers/copycat-gpu-cuda10.2-cudnn7.sif \
sh run_decode_slurm_hands.sh 
