#/bin/bash

if [[ $1 = "reb" ]]; then
scp -oProxyJump=$2@130.226.142.166 $3 $2@10.1.1.121:.
fi