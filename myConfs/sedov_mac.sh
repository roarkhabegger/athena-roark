python3 configure.py \
 --prob=wind \
 --coord=spherical_polar \
 --flux=roe \
 --nghost=2 \
 --eos="adiabatic" \
 -hdf5 \
 -mpi \
 --hdf5_path="/usr/local/hdf5-1.14.0" \
 --cflag='-DH5_HAVE_PARALLEL ' \
