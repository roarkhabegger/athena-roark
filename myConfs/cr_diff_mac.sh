python3 configure.py \
 --prob=cr_diffusion \
 --nghost=2 \
 -b \
 -cr \
 -hdf5 \
 -mpi \
 --hdf5_path="/usr/local/hdf5-1.14.0" \
 --cflag='-DH5_HAVE_PARALLEL'
