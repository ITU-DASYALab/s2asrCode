#!/bin/bash

#SBATCH -J Librispeech
#SBATCH -o ./logs/FE/FE-Librispeech-%J-%N.out
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -c 32
#SBATCH --mem=90000M

python3 src/FE.py \
    --output_folder="FE_data/LibriSpeech/" \
    --dictionary="EN_chars" \
    --amplitude_normalization="False" \
    --include_unknown="False" \
    --input_dataset="data/LibriSpeech" \
    --libri_speech=True \
