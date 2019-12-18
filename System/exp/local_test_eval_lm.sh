#/bin/bash


model="LogisticModel"

python3 src/AM_eval.py \
    --word_based=False \
    --batch_size=16 \
    --num_gpu=0 \
    --LSTM_size=80 \
    --conv_output=256 \
    --conv_width=11 \
    --max_frames=1000 \
    --LSTM_Layer_count=1 \
    --model=$model \
    --curriculum_learning=False \
    --input_tfrecord="FE_data/LibriSpeech/train0000*.tfrecord" \
    --dictionary="EN_chars" \
    --include_unknown=False \
    --summary_name="eval" \
    --custom_beam_search=True \
    --beam_width=8 \
    --include_batch_summary=True \
    --real_language_model=True \
    #--wait_for_checkpoint=False \

