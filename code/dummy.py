# dummy script for testing running jobs on multiple arrays
#%%
import os
import utils
slurm_task_id=os.environ["SLURM_ARRAY_TASK_ID"]
print(slurm_task_id)
print("IN SCRIPT WD:", os.getcwd())
# print(os.environ["SLURM_ARRAY_TASK_ID"],type(array_num))
# subj=utils.assign_subj(slurm_task_id)
# print(f"array num {slurm_task_id}, subject: {subj}")