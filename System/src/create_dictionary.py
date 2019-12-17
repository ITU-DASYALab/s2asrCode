from collections import defaultdict
import string
import numpy as np
import tensorflow as tf
import csv
import json
import shutil
import sys
import os
import re

import util

import argparse

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("--tsv", type=str, default="./data/sv-SE/train.tsv", help="The datafile containing the sentences")
parser.add_argument("--target",type=str, default="SE", help="the outputfile")
parser.add_argument("--mozilla", type=bool, default=True , help= "true if the dataset used is Mozilla Common Voice and false if it is Libri Speech" )

args = parser.parse_args()

def save_dict(target, dictionary):
    with open(target, 'w+', encoding='utf-8') as output_file:
        for x in dictionary.keys():
            output_file.write(x+ " \n")

def save_lines(target, lines):
    with open(target, 'w+', encoding='utf-8') as output_file:
        for x in lines:
            output_file.write(x+ "\n")

def dictionary_mozilla_common_voice(from_path):
    csv_data = list(util.load_csv(from_path,split="\t"))[1::]
    dictionary = defaultdict(lambda: len(dictionary))
    for line in csv_data:
        split = list(filter(None,re.split(r'[^\w]', line[2].lower())))
        for x in split:
            dictionary[x]
    return dictionary

def dictionary_libri_speech():

    test_train_dev = {
        "train-100": "train-clean-100",
        "train-360": "train-clean-360",
        "train-500": "train-other-500",
    }
    dictionary = defaultdict(lambda: len(dictionary))
    lines = []
    for key in test_train_dev:
        base_path = os.path.join("data/LibriSpeech/" , test_train_dev[key], "LibriSpeech", test_train_dev[key])
        dir_list = os.listdir(base_path)
        data = []
        for x in dir_list:
            sub_path = os.path.join(base_path, x)
            sub_dir_list = os.listdir(sub_path)
            for y in sub_dir_list:
                sub_sub_path = os.path.join(sub_path, y)
                sub_sub_dir_list = os.listdir(sub_sub_path)
                txt_file = list(filter(lambda x: "trans" in x, sub_sub_dir_list))
                assert len(txt_file) == 1
                with open(os.path.join(sub_sub_path,txt_file[0]), "r", encoding="utf-8") as f:
                    for line in f:
                        line_split = line.lower().strip().split(" ",1)
                        lines.append(line_split[1])
                        for x in line_split[1].split(" "):
                            word = ''
                            for char_index, char in enumerate(x):
                                if char_index > 0 and char == x[char_index -1]:
                                    word += '´'
                                word += char
                            dictionary[word]

    return dictionary, lines





if __name__ == '__main__':
    from_path = args.tsv
    target_file = "./dict/" + args.target + "_word.txt"
    print(target_file)
    if args.mozilla:
        dictionary = dictionary_mozilla_common_voice(from_path)
        save_dict(target_file, dictionary)
    else: 
        dictionary, lines = dictionary_libri_speech()
        save_dict(target_file,dictionary)
