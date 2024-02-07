#!/bin/bash
##SCRIPT NOTES:


## bash script for segmenting and preprocessing eeg data
#SBATCH -t 01:00:00
#SBATCH --mem=60gb
#SBATCH -n 20 ##Number of tasks
#SBATCH -J debug_preprocess_segment
#SBATCH --partition=debug
#SBATCH --output=clean_wavs_debug_preprocess_segment.log

##SBATCH --array=1-26 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu


##bash vars
which_stmps="xcorr"
script_name="preprocess_segment"
subjs=(
    3253  
    ## 3316  
    ## 3317  
    ## 3318 
    ## 3322  
    ## 3323  
    ## 3325  
    ## 3326
    ## 0194  
    ## 2588  
    ## 2621  
    ## 2782
    ## 3133  
    ## 3146  
    ## 3218  
    ## 3287  
    ## 3314  
    ## 3315  
    ## 3324  
    ## 3328
)
source /scratch/apalaci6/miniconda3/bin/activate lalor0
date
echo "segmenting using $which_stmps"
for subj_num in "${subjs[@]}"
do
    export subj_num
    export which_stmps
    echo "running subject $subj_num"
    python /scratch/apalaci6/columbia_sz/code/$script_name.py
done



