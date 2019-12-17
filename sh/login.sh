#/bin/bash

if [[ $1 = "reb" ]]; then
ssh -J $2@130.226.142.166 $2@10.1.1.121 -t 'cd /mnt/sdb/StreamSpeechV2;bash -l'
elif [[ $1 = "sim" ]]; then
ssh $2@130.226.140.47 -t 'cd /mnt/sdb/S2;bash -l'
elif [[ $1 = "hpc" ]]; then
ssh $2@hpc.itu.dk -t 'cd StreamSpeechV2;bash -l'
fi