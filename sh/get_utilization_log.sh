#/bin/bash

if [ ! -d System/gpu_utilization_log ]; then
    mkdir -p System/gpu_utilization_log;
fi
scp sebab@hpc.itu.dk:StreamSpeechV2/System/gpu_utilization_log/* ./System/gpu_utilization_log/
