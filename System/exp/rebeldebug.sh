#!/bin/bash
name=cudnnlstm2
LSTM_size=1024

python3 src/AM.py \
    --num_gpu=4 \
    --model=StreamSpeechM33Cudnn \
    --batch_size=32 \
    --LSTM_size=$LSTM_size \
    --input_tfrecord="FE_data/EN/train*.tfrecord" \
    --dictionary="EN" \
    --training_directory="models/$name/" \
    &

sleep 16

python3 src/AM_eval.py \
    --model=StreamSpeechM33Cudnn \
    --batch_size=4 \
    --input_tfrecord="FE_data/EN/dev*.tfrecord" \
    --dictionary="EN" \
    --LSTM_size=$LSTM_size \
    --training_directory="models/$name/" \
    --summary_name="dev_data" \
    &


python3 src/AM_eval.py \
    --model=StreamSpeechM33Cudnn \
    --batch_size=4 \
    --input_tfrecord="FE_data/EN/train0000*.tfrecord" \
    --dictionary="EN" \
    --LSTM_size=$LSTM_size \
    --training_directory="models/$name/" \
    --summary_name="subset_data" \
    &

wait