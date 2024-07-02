#!/bin/bash
##SCRIPT NOTES:


## bash script for segmenting and preprocessing eeg data
#SBATCH -t 01:00:00
#SBATCH --mem=60gb
#SBATCH -n 20 ##Number of tasks
#SBATCH -J evnt_resegment_750_3318
#SBATCH --partition=standard
#SBATCH --output=preprocess_with_decimate_pt2.log

##SBATCH --array=1-26 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu 

##bash vars
echo "segmenting eeg using evnt timestamps and overall confidence threshold at 0.75. Updated downsapling w scipy decimate instead of resample. redoing subject 3318 because suspicious results"
do_avg_ref="true"
which_stmps="evnt"
script_name="preprocess_segment"
which_xcorr="wavs"  
noisy_or_clean="clean"
just_stmp="false"
evnt_thresh=0.75
subjs=(
3318
)
##NOTE 0194, 3244, 3283 all done manually during interactive session while debugging script so not included here
source /scratch/apalaci6/miniconda3/bin/activate lalor0
date
echo "segmenting using $which_stmps and $noisy_or_clean $which_xcorr"
echo "just_stmp set to $just_stmp"
for subj_num in "${subjs[@]}"
do
    export subj_num
    export which_stmps
    export which_xcorr
    export noisy_or_clean
    export just_stmp
    export do_avg_ref
    export evnt_thresh
    echo "running subject $subj_num"
    python /scratch/apalaci6/columbia_sz/code/$script_name.py
done
echo "all subjects complete. hooray."


