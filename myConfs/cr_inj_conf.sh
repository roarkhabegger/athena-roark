python3 configure.py \
 --prob=cr_inj \
 --nghost=2 \
 --eos="adiabatic" \
 -b \
 -cr \
 -hdf5 \
 -mpi \
 --mpiccmd=h5pcc \
 --cflag='-DH5_HAVE_PARALLEL -lstdc++' \
