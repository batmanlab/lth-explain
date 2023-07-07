#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/lth_pcbm_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/HAM10k/lth_pcbm_$CURRENT.out
echo "PCBM"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7
# module load cuda/10.1
#python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pcbm.py --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_cub.yaml'> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_pcbm.py --dataset 'HAM10k' > $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_get_concepts.py --dataset 'HAM10k' > $slurm_output