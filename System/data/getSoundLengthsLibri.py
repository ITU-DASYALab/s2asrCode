#/usr/bin/python3


import os
from datetime import datetime, timedelta
import tqdm
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("--part", type=str, default="test-clean", 
    help="the partition")

args = parser.parse_args()

values = []


file = args.part
base_path = os.path.join(
    "LibriSpeech", args.part, 
    "LibriSpeech", args.part)

dir_list = os.listdir(base_path)
values = []
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
				file_path = os.path.join(sub_sub_path,line_split[0]+ ".flac")
				if not  os.path.exists(file_path):
					print("path: " + file_path)
					assert os.path.exists(file_path)
				if not len(line_split[1]) > 0:
					print("path: " + file_path)
					assert len(line_split[1]) > 0
				line_split[0] = file_path
				features= ["", file_path, line_split[1]]
				values.append(features)


values = values[1:]


def audio_time(x):
    file_string = x[1]
    duration_string = os.popen('ffmpeg -i ' + file_string + ' 2>&1 | grep Duration').read()
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

with open("LibriSpeech/" + file + ".durations.csv" , "w+", encoding="utf-8") as f:
    for x in durations:
        f.write(x + "\n")
