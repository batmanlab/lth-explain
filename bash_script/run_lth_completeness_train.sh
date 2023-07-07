#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/lth_completeness_train_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/lth_completeness_train_$CURRENT.out
echo "Training G with pruning for completeness calculation"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7
# module load cuda/10.1
python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_train_completeness.py --dataset 'HAM10k'> $slurm_output