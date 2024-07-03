#!/bin/bash
##SCRIPT NOTES:


## bash script for segmenting and preprocessing eeg data
#SBATCH -t 10:00:00
#SBATCH --mem=60gb
#SBATCH -n 22 ##Number of tasks
#SBATCH -J evnt_resegment_000
#SBATCH --partition=standard
#SBATCH --output=/dev/null

#SBATCH --array=1-22 
##SBATCH --depend=afterany:17146968
#SBATCH --mail-type=END
#SBATCH --mail-user=apalaci6@ur.rochester.edu 
script_name="preprocess_segment" ##name of python script to run
output_dir="LOGS/$script_name"
mkdir -p $output_dir
exec > "$output_dir/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
date
hostname

##bash vars
echo "thresh 0. Updated downsapling w scipy decimate instead of resample."
date
echo "segmenting using $which_stmps and $noisy_or_clean $which_xcorr"
echo "just_stmp set to $just_stmp"

export subj_num="true"
export which_stmps="evnt"
export which_xcorr="wavs" ##I think does nothing?
export noisy_or_clean="clean" ##I think does nothing?
export just_stmp="false"
export do_avg_ref="true"
export evnt_thresh=0.0
source /scratch/apalaci6/miniconda3/bin/activate lalor0

python /scratch/apalaci6/columbia_sz/code/$script_name.py


