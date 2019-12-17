#/bin/bash

if [[ $1 = "reb" ]]; then
echo "Connected to RebelRig. Open localhost:$3 on your local machine."
ssh -N -L $3:rebelrig:$3 -J $2@130.226.142.166 $2@10.1.1.121 
echo "Closed connection to RebelRig."
elif [[ $1 = "hpc" ]]; then 
echo "Connected to HPC"
ssh -N -L $3:front:$3 $2@hpc.itu.dk 
echo "Closed connection"
fi