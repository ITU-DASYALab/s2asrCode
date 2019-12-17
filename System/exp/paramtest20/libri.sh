#!/bin/bash

#SBATCH -J libri-p20
#SBATCH -o ./logs/libri-p20-%A-%a-%N.out
#SBATCH -N 1
#SBATCH --array=0-19%1
#SBATCH -t 12:00:00
#SBATCH -c 48
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=190000M


echo "Running on: $(hostname)"

batch_size=300
eval_batch_size=150
LSTM_size=800
LSTM_Layer_count=5
conv_output=600
conv_width=11
max_frames=1670
dropout=0.05
beam_width=1

input_tfrecord="FE_data/LibriSpeech/train*.tfrecord"
input_tfrecord_eval="FE_data/LibriSpeech/dev*.tfrecord"
dictionary="EN_chars"
model="StreamSpeechM33"



activation_function="relu"
num_epochs=1

if (( SLURM_ARRAY_TASK_ID > 0 )); then
    new_model=False
else
    new_model=True
fi

name=$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID
paramnum="para20"
training_directory="models/$paramnum/$name/"


params="--LSTM_size=$LSTM_size \
    --LSTM_Layer_count=$LSTM_Layer_count \
    --conv_output=$conv_output \
    --conv_width=$conv_width \
    --dictionary=$dictionary \
    --curriculum_learning=False \
    --include_unknown=False \
    --max_frames=$max_frames \
    --model=$model \
    --activation_function=$activation_function \
    --l2_batch_normalization=False "

iteration=1
while true; do 


python3 src/AM.py \
    --training_directory=$training_directory \
    --input_tfrecord=$input_tfrecord \
    --buffer_size=100 \
    --num_epochs=$num_epochs \
    --new_model=$new_model \
    --num_gpu=2 \
    --dropout=$dropout \
    --batch_size=$batch_size \
    --custom_beam_search=False \
    --beam_width=1 \
    --num_parallel_reader=16 \
    --drop_remainder=True \
    --seed=$iteration \
    $params \

python3 src/AM_eval.py \
    --training_directory=$training_directory \
    --input_tfrecord=$input_tfrecord_eval \
    --summary_name="dev-wordCTC" \
    --wait_for_checkpoint=False \
    --num_gpu=1 \
    --batch_size=$eval_batch_size \
    --custom_beam_search=True \
    --beam_width=64 \
    $params \

new_model=False
iteration=$((iteration+1))
done
