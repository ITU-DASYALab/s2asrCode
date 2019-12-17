#/usr/bin/python3


import os
from datetime import datetime, timedelta
import tqdm
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("--tsv", type=str, default="EN/train.tsv", help="The datafile containing the sentences")

args = parser.parse_args()

values = []
file = args.tsv
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        values.append([x for x in line.strip().split("\t")])

values = values[1:]

def audio_time(values):
    file_string = values[1]
    duration_string = os.popen('ffmpeg -i EN/clips/' + file_string + ' 2>&1 | grep Duration').read()
    audio_timestamp = duration_string.split(" ")[3][:-1].replace('.',',').ljust(15, '0')
    if "N/A" in audio_timestamp:
        mili_seconds = 0.0
    else:
        mili_seconds = (datetime.strptime(audio_timestamp, '%H:%M:%S,%f') - \
            datetime.strptime('00', '%H')).total_seconds()*1000
    return file_string +","+  str(mili_seconds)

pool = mp.Pool(mp.cpu_count())

durations = []
for x in tqdm.tqdm(pool.imap_unordered(audio_time, values), total=len(values)):
    durations.append(x)

pool.close()
pool.join()

with open(file + ".durations.csv" , "w+", encoding="utf-8") as f:
    for x in durations:
        f.write(x + "\n")
