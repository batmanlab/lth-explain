#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/HAM10k_prune_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/HAM10k_prune_$CURRENT.out
echo "Pruning-LTH"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

# For datasets other than HAM10k
#conda activate python_3_7_rtx_6000

# For HAM10k
conda activate python_3_7
# module load cuda/10.1
# python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pruning.py --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_mnist.yaml'> $slurm_output
# python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pruning.py --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_cub_resnet_101.yaml'> $slurm_output

#python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pruning.py --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_derma.yaml'> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pruning.py --dataset 'HAM10k'> $slurm_output