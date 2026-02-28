#!/bin/sh 
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --output=/beegfs/%u/codes/shell/submit/jnb.job.%j
#SBATCH --partition=gpu
#SBATCH --job-name=SAM3

source ~/.bashrc

conda activate geo 
cd /beegfs/halder/GITHUB/RESEARCH/WBCrop/src/00_data_preparation

python 01_segment_field_using_SAM3.py >& field_segmentation_log.txt
