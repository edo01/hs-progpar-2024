#!/bin/bash

KERNEL="blur2" # mandel
WT="urrot2_neon_div8_u16" #default

echo "######################################"
echo "Exploring the best parameters for the kernel $KERNEL"
echo "using $WT"
echo "######################################"
# this script allows you to explore different tiles dimensions, 
# number of threads and omp scheduling types

# since the do_tile function has the same complexity for all the
# pixels, we assume that the static scheduling is the best option

# we will explore the following parameters:
# - full horizontal tiling:  

# print the header of the csv file
echo "th,tw,threads,schedule,schedule_n,time"

for RUNTIME in "static,1" "static,2" "static,3"; do
    for TH in 64 128 256 512 1024; do
        for TW in 64 128 256 512 1024; do
            # skip 1024x1024
            if [ $TH -eq 1024 ] && [ $TW -eq 1024 ]; then
                continue
            fi

            NTILES=$((1024/$TH * 1024/$TW))
            # CORTEX
            for THREADS in 1 2 3 4; do
                # skip if the number of tiles is less than the number of threads
                if [ $NTILES -lt $THREADS ]; then
                    continue
                fi 
                export OMP_NUM_THREADS=$THREADS
                export OMP_SCHEDULE=$RUNTIME
                # to correctly set the affinity we need to know the number of threads
                let UB=$THREADS+1
                RANGE=0$(test $THREADS -gt 1 && echo -n "," && seq -s',' 3 $UB )
                TIME=$(taskset -c $RANGE ./run.sh -k $KERNEL -l images/1024.png  -v omp_tiled -wt $WT -th $TH -tw $TW -n -si 2>&1| tail -1)
                # print the results in a csv format
                echo "cortex",$TH,$TW,$THREADS,$RUNTIME,$TIME
            done
            # DENVER
            for THREADS in 1 2; do
                # skip if the number of tiles is less than the number of threads
                if [ $NTILES -lt $THREADS ]; then
                    continue
                fi 
                export OMP_NUM_THREADS=$THREADS
                export OMP_SCHEDULE=$RUNTIME
                RANGE=1$(test $THREADS -gt 1 && echo -n ",2")
                TIME=$(taskset -c $RANGE ./run.sh -k $KERNEL -l images/1024.png  -v omp_tiled -wt $WT -th $TH -tw $TW -n -si 2>&1| tail -1)
                # print the results in a csv format
                echo "denver",$TH,$TW,$THREADS,$RUNTIME,$TIME
            done
            # full system
            # DENVER
            THREADS=6
            # skip if the number of tiles is less than the number of threads
            if [ $NTILES -lt $THREADS ]; then
                continue
            fi 
            export OMP_NUM_THREADS=$THREADS
            export OMP_SCHEDULE=$RUNTIME
            RANGE=$(seq -s',' 0 5)
            TIME=$(taskset -c $RANGE ./run.sh -k $KERNEL -l images/1024.png  -v omp_tiled -wt $WT -th $TH -tw $TW -n -si 2>&1| tail -1)
            # print the results in a csv format
            echo "full_system",$TH,$TW,$THREADS,$RUNTIME,$TIME
        done
    done
done 