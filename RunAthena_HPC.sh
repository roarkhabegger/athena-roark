#!/bin/bash
#SBATCH --mem-per-cpu=1000  #<- 256GB per node, 128 cores per node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --partition=shared     
#SBATCH --job-name=test2
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
spack env activate athenaICM

srun ./athena -i athinput.realistic_grav -t 23:50:00 > silcc.log
