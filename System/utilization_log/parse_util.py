import numpy as np
import csv
import argparse

def parse(file, nrGPU):
    total_line_nr = 0
    line_nr = 0
    contents = []
    firstTime = 0
    gpu_vals = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if line_nr == 0:
                    time = int(line.strip())
                    if firstTime == 0:
                        firstTime = time
                elif line_nr == 1:
                    cpu = line.strip()
                else:
                    gpu_vals.append(int(line.strip().split(",")[1].replace('%','').strip()))
                line_nr += 1
                total_line_nr += 1
                if line_nr == 2 + nrGPU:
                    row = [time - firstTime, float(cpu)]
                    row.extend(gpu_vals)
                    contents.append(row)
                    gpu_vals = []
                    line_nr = 0
            except IndexError:
                print(line)
                print(total_line_nr)
                break
    return contents


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--data", type=str, default="HPC-2019-10-20_15:18:57.data", 
        help="The datafile containing the measurements")
    parser.add_argument("--gpus",type=int, default=2, help="the number of gpus")

    args = parser.parse_args()

    file_in = args.data
    data = parse(file_in, args.gpus)

    columns = ['Time','CPU']
    for x in range(args.gpus):
        columns.append("GPU" + str(x + 1))

    with open(file_in.replace('.data','.csv'), "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(columns)
        csv_writer.writerows(data)

    data_structured = np.array(data)


