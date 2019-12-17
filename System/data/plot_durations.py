#/usr/bin/python3


import os
from datetime import datetime, timedelta
import tqdm
import multiprocessing as mp
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("--tsv", type=str, default="EN/train.tsv.durations.csv", help="The datafile containing the sentences")

args = parser.parse_args()

values = []
file = args.tsv
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        values.append([x for x in line.strip().split(",")])

durations = [float(x[1]) for x in values]
durations.sort()


#plt.hist(durations, 100)

#plt.show()
def mili_seconds_to_time(milisec):
    milisec_rounded = int(milisec) % 1000
    seconds = int(milisec / 1000)
    seconds_rounded = seconds % 60
    minutes = int(seconds / 60)
    minutes_rounded = minutes % 60
    hours = int(minutes / 60)

    return "hour %d, min %d, sec, %d mil, %d" % (hours, minutes_rounded, seconds_rounded, milisec_rounded)

print("total", mili_seconds_to_time(np.sum(durations)))

print("mean",np.mean(durations))
print("median",np.median(durations))

print("quantile 25",np.quantile(durations,0.25))
print("quantile 75",np.quantile(durations,0.75))
print("quantile 90",np.quantile(durations,0.90))
print("quantile 95",np.quantile(durations,0.95))
print("quantile 98",np.quantile(durations,0.98))
print("quantile 99",np.quantile(durations,0.99))
print("quantile 99.9",np.quantile(durations,0.999))

print("longest",durations[-1])
print("longest", mili_seconds_to_time(durations[-1]))
print("longest-1",durations[-2])
print("longest-2",durations[-3])

