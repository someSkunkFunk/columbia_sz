#!/bin/bash
##SCRIPT NOTES:


## bash script for segmenting and preprocessing eeg data
#SBATCH -t 04:00:00 ## max time needed using nested crossval
#SBATCH --mem=60gb
#SBATCH -n 20 ##Number of tasks
#SBATCH -J ed_env_recon
#SBATCH --partition=standard
#SBATCH --output=ed_env_recon.log

##SBATCH --array=1-26 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu

script_name="ed_env_recon" ##name of python script to run

source /scratch/apalaci6/miniconda3/bin/activate lalor0
date
hostname
##NOTE: spaces important for string comparisons

python /scratch/apalaci6/columbia_sz/code/$script_name.py
echo "all trfs complete."
