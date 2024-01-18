python3 configure.py \
 --prob=orszag_tang_outflow \
 --nghost=2 \
 -b \
 -hdf5 \
 -mpi \
 --hdf5_path="/usr/local/hdf5-1.14.0" \
 --cflag='-DH5_HAVE_PARALLEL'
