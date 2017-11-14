#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=AlexNet
#SBATCH --time=24:00:00
#SBATCH --mem=8GB


module purge
module load cuda/8.0.44
module load torch/intel/20170104
module load python/intel/2.7.12
module load pytorch/intel/20170724
module load torchvision/0.1.7

python AlexNet.py >>out_AlexNet.log 2>&1
