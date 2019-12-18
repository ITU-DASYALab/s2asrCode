#!/bin/bash

#SBATCH -J FE
#SBATCH -o ./logs/FE/FE-%A-%a-%N.out
#SBATCH --array=0-2
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -c 48
#SBATCH --mem=90000M
         

echo "Running on: $(hostname)"
echo $SLURM_ARRAY_TASK_ID

output_folder="FE_data/EN/FE_Chars/"
input_dataset="data/EN/"
dictionary="EN_chars"

data=("dev" "test" "train" )
data_sets="${data[$SLURM_ARRAY_TASK_ID]}"

echo $output_folder 
echo $input_dataset 
echo $dictionary
echo $data_sets

python3 src/FE.py \
    --output_folder=$output_folder \
    --input_dataset=$input_dataset \
    --dictionary=$dictionary \
    --data_sets=$data_sets \
    --amplitude_normalization="False" \
    --include_unknown="False" \
    --parallelism_degree=24

