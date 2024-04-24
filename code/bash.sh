#!/bin/bash
##SCRIPT NOTES:


## template bash script

#SBATCH -t 0-01:00:00
##SBATCH --mem=60gb
##SBATCH -n 20 ##Number of tasks
#SBATCH -J dum
#SBATCH --partition=debug
#SBATCH --array=1-20
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu


##set up output logs 
#SBATCH --output=/dev/null
script_name='dummy' ##name of python script to run
output_dir="LOGS/$script_name"
mkdir -p $output_dir
exec > "$output_dir/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
date
hostname
## Variables for importing into python

## starts python (using conda environment) and executes the script


source /scratch/apalaci6/miniconda3/bin/activate lalor0
python /scratch/apalaci6/columbia_sz/code/$script_name.py
