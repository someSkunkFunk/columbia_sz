#!/bin/bash
##SCRIPT NOTES:


## bash script for backwards trf using nested cv
#SBATCH -t 2-00:00:00
#SBATCH --mem=60gb
#SBATCH -n 22 ##Number of tasks
#SBATCH -J block6_fwd_model
#SBATCH --partition=standard


#SBATCH --array=1-22
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu
##set up output logs 
#SBATCH --output=/dev/null
script_name="og_env_recon" ##name of python script to run
output_dir="LOGS/$script_name"
mkdir -p $output_dir
exec > "$output_dir/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
date

hostname

echo "forward model with nested cross validation"
export direction="1"
export which_stmps="evnt"
export which_xcorr="wavs"
export evnt_thresh="750"
export k_folds="5"
export shuffle_trials="true"
export blocks="6"
## export blocks="1,2,3,4,5"
## export blocks="all"
export cv_method='nested'
export which_envs="rms"

source /scratch/apalaci6/miniconda3/bin/activate lalor0

##NOTE: spaces important for string comparisons
if [ $which_stmps = "xcorr" ]; then
    echo "timestamps from xcorr - $which_xcorr used"
elif [ $which_stmps = "evnt" ]; then
    echo "timestamps from evnt used"
fi

python /scratch/apalaci6/columbia_sz/code/$script_name.py

