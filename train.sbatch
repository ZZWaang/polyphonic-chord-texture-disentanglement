#!/bin/bash
#SBATCH --job-name=polydis2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=polydis2.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ly1387@nyu.edu

module purge
module load anaconda3/2020.07

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

conda activate /scratch/ly1387/polydis2;
export PATH=/scratch/ly1387/polydis2/bin:$PATH;

cd /scratch/ly1387/projects/polydis2/polydis2;
python train.py --epoch 12 --parallel True --training_stage 2
