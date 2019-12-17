
echo "Running on: $(hostname)"

batch_size=96
eval_batch_size=24
LSTM_size=800
LSTM_Layer_count=5
conv_output=600
conv_width=11
max_frames=1670
dropout=0.05

input_tfrecord="FE_data/LibriSpeech/train*.tfrecord"
input_tfrecord_eval="FE_data/LibriSpeech/dev*.tfrecord"
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

iteration=1
while true; do 


python3 src/AM.py \
    --training_directory=$training_directory \
    --input_tfrecord=$input_tfrecord \
    --buffer_size=40 \
    --num_epochs=$num_epochs \
    --new_model=$new_model \
    --num_gpu=4 \
    --dropout=$dropout \
    --batch_size=$batch_size \
    --custom_beam_search=False \
    --beam_width=1 \
    --num_parallel_reader=8 \
    --drop_remainder=True \
    --seed=$iteration \
    $params \

python3 src/AM_eval.py \
    --training_directory=$training_directory \
    --input_tfrecord=$input_tfrecord_eval \
    --summary_name="dev" \
    --wait_for_checkpoint=False \
    --num_gpu=1 \
    --batch_size=$eval_batch_size \
    --custom_beam_search=True \
    --beam_width=64 \
    $params \

new_model=False
iteration=$((iteration+1))
done
