#!/bin/bash
#SBATCH --mem=32g  #<- 256GB per node, 128 cores per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu     
#SBATCH --account=bdru-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=ath_make
#SBATCH --time=00:10:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out


cd /u/roark/athena-roark/
module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load openmpi hdf5 fftw
make clean
make clean

sh myConfs/cr_icm_delta.sh

make -j 32
