#/bin/bash


## The process to monitor locally
process_id=$1

## The name to prepend to the output file
Base_LOGNAME=`date '+%F_%H:%M:%S'`
if [ "$2" != "" ]; then
    LOGNAME="$2-$Base_LOGNAME"
else
    LOGNAME=$Base_LOGNAME
fi

## The interval at which to get utilization
INTERVAL=0.01
if [ "$3" != "" ]; then
    INTERVAL="$3"
fi


while true
do 
    date +%s%N | cut -b1-13 >> "$LOGNAME".data
    top -n 1 -d 0.03 -p $process_id | grep $process_id | grep -o 'S \+[0-9.]\+' | cut -c 3- >> "$LOGNAME".data
    #ps -eo pcpu,pmem | sort -k 1 -r | head -2 >> System/gpu_utilization_log/"$LOGNAME".log
    nvidia-smi --format=csv,noheader --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu >> "$LOGNAME".data
    sleep $INTERVAL
done
