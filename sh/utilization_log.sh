#/bin/bash
if [ ! -d System/gpu_utilization_log ]; then
    mkdir -p System/gpu_utilization_log;
fi
Base_LOGNAME=`date '+%F_%H:%M:%S'`
if [ "$2" != "" ]; then
    LOGNAME="$2-$Base_LOGNAME"
else
    LOGNAME=$Base_LOGNAME
fi
INTERVAL=1
if [ "$1" != "" ]; then
    INTERVAL="$1"
fi
process_id=$3 
while true
do 
    date +%s%N | cut -b1-13 >> Res/"$LOGNAME".data
    top -n 1 -d 0.03 -p $process_id | grep $process_id | grep -o 'S \+[0-9.]\+' | cut -c 3- >> Res/"$LOGNAME".data
    #ps -eo pcpu,pmem | sort -k 1 -r | head -2 >> System/gpu_utilization_log/"$LOGNAME".log
    nvidia-smi --format=csv,noheader --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu >> Res/"$LOGNAME".data
    sleep $INTERVAL
done
