#!/bin/bash

python3 src/create_dictionary.py

python3 src/create_dictionary.py \
    --tsv ./data/EN/train.tsv \
    --target EN