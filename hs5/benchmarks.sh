#!/bin/bash

# for each kernel optimization, we run the program multiple times to get the best execution time. 
# In this way we can obtain a more accurate result, since the execution time can vary depending on the system load.

# Then we run the program using different compilation optimization flags to see how compiler optimization affects 
# the execution time.

# Compile the program with different optimization flags

for OPTIMIZATION in "O3"
do
    sed -ie "s/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -$OPTIMIZATION/g" Makefile
    make -j4 1>/dev/null 2>/dev/null
    
    D2=1000000000
    CA57=1000000000

    for i in {0..3} 
    do
        echo "Running kernel with optimaztion $OPTIMIZATION" >&2
        # O0
        D2_c=$(taskset -c 1 ./bin/easypap -k heat --no-display -r 500 -i 5000 --mpirun "-np 2" -si -v mpi_v0 -wt bv2 2>&1| tail -1)
        CA57_c=$(taskset -c 0 ./bin/easypap -k heat --no-display -r 500 -i 5000 --mpirun "-np 2" -si -v mpi_v0 -wt bv2 2>&1| tail -1)

        echo "D2 ($OPTIMIZATION): $D2_c ms" >&2
        echo "CA57 ($OPTIMIZATION): $CA57_c ms" >&2

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_c < ${D2}" | bc) -eq 1 ]
        then
            D2=$D2_c
            echo "update D2_c" >&2
        fi

        if [ $(echo "$CA57_c < ${CA57}" | bc) -eq 1 ]
        then
            CA57=$CA57_c
            echo "update CA57_c" >&2
        fi

    done

    # print the best execution time for each kernel
    echo "-----------------------------------"
    echo "----------OPTIMIZATION $OPTIMIZATION----------"
    echo "-----------------------------------"
    echo 
    echo "D2: ${D2[0]} ms"
    echo "CA57: ${CA57[0]} ms"
    echo
    echo
    echo
done