#!/bin/bash
#SBATCH --mem-per-cpu=4000  #<- 256GB per node, 128 cores per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=pre   
#SBATCH --job-name=ath_make
#SBATCH --time=0-00:10:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out


cd /u/roark/athena-roark/
module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
spack env activate athenaICM
make clean
make clean

# sh myConfs/cr_icm_hpc.sh
# sh myConfs/cr_sne_hpc.sh
sh myConfs/real_grav_conf.sh

make -j 32
