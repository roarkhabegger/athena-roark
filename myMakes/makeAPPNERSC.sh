#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=0-12:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
module load cray-hdf5-parallel/1.12.2.1
module load cray-fftw/3.3.10.2

cd /global/u1/r/roarkh/athena-roark

make clean
make clean
sh myConfs/cr_turb.sh
make -j 32
