#!/bin/bash

# Inspired by https://github.com/baidu-research/ba-dls-deepspeech


find . -iname "*.flac" | wc
index=0
for flacfile in `find . -iname "*.flac"`
do
    ffmpeg -loglevel error -hide_banner -nostats -i $flacfile -y -vn -b:a 64k -ac 1 -ar 16000 "${flacfile%.*}.wav" &
    (( index++ ))
    if (( index % 100 == 99 )); then
        sleep 0.2
    fi
    #echo "$index $flacfile"
done

wait
