#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=AlexNet
#SBATCH --time=24:00:00
#SBATCH --mem=10GB


module purge
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
python3 AlexNet.py >>AlexNet.log 2>&1
