#/bin/bash

python3 src/FE.py \
    --output_folder="FE_data" \
    --test=True \
    --feature_names="power_spectrograms","magnitude_spectrograms","mel_spectrograms","log_mel_spectrograms","all_mfcc","mfccs" \
	--include_org_signal=True \
    --amplitude_normalization=True \

echo "------------------------------"
echo "------------------------------"
echo "------------------------------"
echo "------------------------------"

python3 src/plot_tfrecord.py \
    --image_size=1500,3000 \
    --output_name="FE_full" \
    --feature_names="power_spectrograms","magnitude_spectrograms","mel_spectrograms","log_mel_spectrograms","all_mfcc","mfccs" \
    --feature_sizes="257","257","80","80","80","13" \
    --dpi=200 \


echo "------------------------------"
echo "------------------------------"


python3 src/plot_tfrecord.py \
    --image_size=1500,800 \
    --output_name="FE" \
    --feature_names="log_mel_spectrograms","mfccs" \
    --feature_sizes="80","13" \
    --dpi=200 \

echo "------------------------------"
echo "------------------------------"

python3 src/plot_tfrecord.py \
    --image_size=1500,800 \
    --output_name="FE_modified_mfcc" \
    --feature_names="log_mel_spectrograms","mfccs" \
    --feature_sizes="80","13" \
    --include_first_dim_in_mfcc=False \
    --dpi=200 \