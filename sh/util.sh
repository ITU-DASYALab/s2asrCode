


nvidia-smi --format=csv,noheader --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu
ps -eo pcpu,pmem | sort -k 1 -r | head -2
date +%s%N | cut -b1-13

