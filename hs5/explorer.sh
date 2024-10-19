#!/bin/bash

KERNEL="heat" # mandel
wt="bv2" #default
VERSION="mpi_v1"

echo "######################################"
echo "Exploring the best parameters for the kernel $KERNEL"
echo "using $wt and $VERSION"
echo "######################################"
# this script allows you to explore different tiles dimensions, 
# number of threads and omp scheduling types

# since the do_tile function has the same complexity for all the
# pixels, we assume that the static scheduling is the best option

# we will explore the following parameters:
# - full horizontal tiling:  

# print the header of the csv file
echo "cpu,process,time"

           
# CORTEX
for THREADS in 1 2 4; do
    #export OMP_NUM_THREADS=$THREADS
    #export OMP_SCHEDULE=$RUNTIME
    # to correctly set the affinity we need to know the number of threads
    let UB=$THREADS
    RANGE=0$(test $THREADS -gt 1 && echo -n "," && seq -s',' 1 $THREADS )
    TIME=$(./run -k $KERNEL  -v $VERSION -wt $wt --mpirun "--cpu-set $RANGE --bind-to core --report-bindings -n $THREADS" -r 500 -i 5000 -n -si 2>&1| tail -1)
    # print the results in a csv format
    echo "cortex",$THREADS,$TIME
done

# DENVER
for THREADS in 1 2; do
    #export OMP_NUM_THREADS=$THREADS
    #export OMP_SCHEDULE=$RUNTIME
    RANGE=4$(test $THREADS -gt 1 && echo -n ",5")
    TIME=$(./run.sh -k $KERNEL  -v $VERSION -wt $wt --mpirun "--cpu-set $RANGE --bind-to core --report-bindings -n $THREADS" -r 500 -i 5000 -n -si 2>&1| tail -1)
    # print the results in a csv format
    echo "denver",$THREADS,$TIME
done
# full system
# DENVER
THREADS=4
#export OMP_NUM_THREADS=$THREADS
#export OMP_SCHEDULE=$RUNTIME
TIME=$(./run.sh -k $KERNEL -v $VERSION -wt $wt --mpirun "--cpu-set 0,1,4,5 --bind-to core:overload-allowed --report-bindings -n $THREADS" -r 500 -i 5000 -n -si 2>&1| tail -1)
# print the results in a csv format
echo "full_system",$THREADS,$TIME

THREADS=8
TIME=$(./run.sh -k $KERNEL -v $VERSION -wt $wt --mpirun "--cpu-set 0,1,2,3,4,5 --bind-to core:overload-allowed --report-bindings -n $THREADS" -r 500 -i 5000 -n -si 2>&1| tail -1)
# print the results in a csv format
echo "full_system",$THREADS,$TIME

done