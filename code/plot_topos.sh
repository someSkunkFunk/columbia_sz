#!/bin/bash
##SCRIPT NOTES:


## bash script for plotting topos
#SBATCH -t 01:10:00
#SBATCH --mem=60gb
#SBATCH -n 1 ##Number of tasks
#SBATCH -J plot_topos
#SBATCH --partition=standard
#SBATCH --output=plot_topos.log

##SBATCH --array=1-26 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu

script_name='plot_topos' ##name of python script to run
subjs=(
    3253  
    3316  
    3317  
    3318 
    3322  
    3323  
    3325  
    3326
    0194  
    2588  
    2621  
    2782
    3133  
    3146  
    3218  
    3287  
    3314  
    3315  
    3324  
    3328
)
source /scratch/apalaci6/miniconda3/bin/activate lalor0
for subj_num in "${subjs[@]}"
do
    echo "running subject $subj_num"
    export subj_num
    python /scratch/apalaci6/columbia_sz/code/$script_name.py
    echo "$subj_num topos complete"
done
echo "all subjects complete."