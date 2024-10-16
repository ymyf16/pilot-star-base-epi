#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -p defq
#SBATCH --job-name=benchmark_starbase
#SBATCH -t 5:00:00
#SBATCH --mem=10000MB
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=Yufei.Meng@cshs.com

source /apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate starbase

cd /common/Source
python evolver_uni.py 
