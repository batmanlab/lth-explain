#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/CAV_Train_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/CAV_Train_$CURRENT.out
echo "CAV generation using Pruning"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000
# module load cuda/10.1
#python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_generate_cavs.py  --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_mnist.yaml'> $slurm_output
#python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_generate_cavs.py  --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_cub.yaml'> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_generate_cavs.py  --dataset 'HAM10k' > $slurm_output