python3 configure.py \
 --prob=shock_tube_SSMR \
 --nghost=2 \
 -hdf5 \
 -mpi \
 -debug \
 --hdf5_path="/usr/local/hdf5-1.14.0" \
 --cflag='-DH5_HAVE_PARALLEL'
