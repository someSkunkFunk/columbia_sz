#!/bin/bash
##SCRIPT NOTES:


## bash script for mtrf analysis of columbia sz dataset on bluehive
#SBATCH -t 0-01:00:00
#SBATCH --mem=60gb
#SBATCH -n 20 ##Number of tasks
#SBATCH -J columbia_sz_standard_60gbmem_20tasks
#SBATCH --partition=standard
#SBATCH --output=standard_60gbmem_20tasks.log

##SBATCH --array=1-26 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu

script_name='blue_env_recon' ##name of python script to run
subj_num='3316'
export subj_num
## Variables for importing into python

## starts python (using conda environment) and executes the script


source /scratch/apalaci6/miniconda3/bin/activate lalor0
python /scratch/apalaci6/columbia_sz/code/$script_name.py
