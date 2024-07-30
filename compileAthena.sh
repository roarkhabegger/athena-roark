#!/bin/sh
#This file is called compileAthena.sh
#SBATCH --partition=shared       # default "shared", if not specified
#SBATCH --time=0-00:02:00       # run time in days-hh:mm:ss
#SBATCH --nodes=1               # require 4 nodes
#SBATCH --ntasks-per-node=8    # cpus per node (by default, "ntasks"="cpus")
#SBATCH --mem-per-cpu=4000             # RAM per cpu in megabytes
#SBATCH --error=job/job.%J.err
#SBATCH --output=job/job.%J.out
# Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

# Now list your executable command (or a string of them).
module load hdf5
cd $SLURM_SUBMIT_DIR
make clean
make clean
sh myConfs/ntf_cyl.sh
make -j 8
# Submit code using "sbatch compileAthena.sh"
