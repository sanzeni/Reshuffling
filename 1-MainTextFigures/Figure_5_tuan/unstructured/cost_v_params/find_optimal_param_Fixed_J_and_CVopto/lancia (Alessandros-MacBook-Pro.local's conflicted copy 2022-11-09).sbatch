#!/bin/bash
#SBATCH --mail-user=alessandro.sanzeni@duke.edu
#SBATCH --array 0-800
#SBATCH -t 6:00:00             # max runtime
#SBATCH --mem=1g
#SBATCH -J contrast_response
#SBATCH -o output/%J-%a-o.log
#SBATCH -e output/%J-%a-e.log

#SBATCH -p scavenger

# 2 cpus/task, because of hyperthreading. 
### #SBATCH --cpus-per-task=1
### #SBATCH --ntasks-per-core=1
export MKL_NUM_THREADS=2
export QT_QPA_PLATFORM=offscreen

mkdir -p /work/as974
export TMPDIR=/work/as974

### source ~/anaconda3/bin/activate brian2_env

echo $PYTHONPATH
#export PYTHONPATH=$HOME/Documents/RemoteIpythonModules/150108SimInBalanced:$HOME/Documents/RemoteIpythonModules/packages

# start job
python -u sim.py -p params_$SLURM_ARRAY_TASK_ID.txt \
-r results_$SLURM_ARRAY_TASK_ID.txt \
-J $SLURM_ARRAY_TASK_ID


