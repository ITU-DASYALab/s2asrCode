#/bin/bash

model="LogisticModel"

new_model=True

python3 src/AM.py \
    --batch_size=128 \
    --num_gpu=1 \
    --LSTM_size=80 \
    --conv_output=256 \
    --conv_width=11 \
    --max_frames=1600 \
    --LSTM_Layer_count=1 \
    --model=$model \
    --eval_steps=10 \
    --save_model_interval=300 \
    --curriculum_learning=False \
    --steps_until_max_frames=1000 \
    --dropout=0.05 \
    --input_tfrecord="FE_data/LibriSpeech/train0000*.tfrecord" \
    --dictionary="EN_chars" \
    --include_unknown=False \
    --new_model=$new_model \
    --custom_beam_search=True \
    --beam_width=8 \
    --num_parallel_reader=4 \
    --buffer_size=4 \
    #--automatic_mixed_precision=True \
