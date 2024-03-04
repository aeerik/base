#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-752 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1  # launch 1 node 2 T4s
#SBATCH -t 0-10:00:00

echo "Initializing run..."
apptainer exec ~/vm/base.sif python -u ~/base/run.py #mitt jobb är skickat och python kör -u är unbuffered för print
echo "Run complete."S
