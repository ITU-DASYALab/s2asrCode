#/bin/bash

if [[ $1 = "reb" ]]; then
scp -oProxyJump=$2@130.226.142.166 -r ./System/src/*.py $2@10.1.1.121:/mnt/sdb/StreamSpeechV2/System/src/
scp -oProxyJump=$2@130.226.142.166 -r ./System/src/exp $2@10.1.1.121:/mnt/sdb/StreamSpeechV2/System/src/

ssh -J $2@130.226.142.166 $2@10.1.1.121 > /dev/null 2>&1 << EOF
cd /mnt/sdb/StreamSpeechV2/System/src 
chown $(whoami):S2 -R *.py 
chmod 770 -R *.py 
chown $(whoami):S2 -R exp/*.sh 
chmod 770 -R exp/*.sh 
EOF

elif [[ $1 = "sim" ]]; then
echo "TODO"
elif [[ $1 = "hpc" ]]; then
echo "TODO"
fi