#!/bin/sh
#SBATCH --time=00:10:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=128
#SBATCH --job-name=ar_comp  # job name
#SBATCH --output=job.%j.out   # output file name
#SBATCH --error=job.%j.err   # job name
#SBATCH --account=w26_supert   # account name
#SBATCH --qos=standard
#SBATCH --no-requeue   # do not requeue when preempted and on node failure
#SBATCH --signal=23@60  # send signal to job at [seconds] before end
#SBATCH --hint=nomultithread


module reset

module load PrgEnv-gnu
module load cray-mpich
module load cray-hdf5-parallel cray-fftw


cd /users/roarkh/athena-roark/

make clean
make clean

sh myConfs/smallBox_chic.sh

make -j 128
