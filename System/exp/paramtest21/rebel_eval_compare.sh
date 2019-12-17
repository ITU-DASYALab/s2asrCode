
echo "Running on: $(hostname)"

eval_batch_size=24
LSTM_size=800
LSTM_Layer_count=5
conv_output=600
conv_width=11
max_frames=1670
dropout=0.05

input_tfrecord_eval="FE_data/LibriSpeech/dev0000*.tfrecord"

dictionary="EN_chars"

model="StreamSpeechM33"

activation_function="relu"

num_epochs=1

new_model=True


name="Rebel-5"
training_directory="models/$name/"


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

if [ $2 = "t" ]; then
    custom_beam_search=1
else
    custom_beam_search=0
fi

if [ $3 = "t" ]; then
    custom_beam_search=0
    real_language_model=1
else
    real_language_model=0
fi

# PT = ParameterTests
python3 src/AM_eval.py \
    --training_directory=$training_directory \
    --input_tfrecord=$input_tfrecord_eval \
    --summary_name="PT-$1-$2-$3-$4-$5-$6-$7-$8-$9-${10}" \
    --wait_for_checkpoint=False \
    --num_gpu=1 \
    --batch_size=${10} \
    --custom_beam_search=$custom_beam_search \
    --beam_width=$4 \
    --include_batch_summary=True \
    --real_language_model=$real_language_model \
    --lm_beam_width=$5 \
    --cutoff_top_n=$8 \
    --cutoff_prob=$9 \
    --lm_beta=$6 \
    --lm_alpha=$7 \
    $params \

new_model=False
