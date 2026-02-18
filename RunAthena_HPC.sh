#!/bin/bash
#SBATCH --mem-per-cpu=4000  #<- 256GB per node, 128 cores per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=shared     
#SBATCH --job-name=ath_make
#SBATCH --time=0-24:00:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
spack env activate athenaICM
make clean
make clean

srun ./athena -i athinput.cr_sne -t 0-23:50:00 > athena.01.log
