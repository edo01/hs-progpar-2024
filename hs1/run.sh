#!/bin/bash

# for each kernel optimization, we run the program multiple times to get the best execution time. 
# In this way we can obtain a more accurate result, since the execution time can vary depending on the system load.

# Then we run the program using different compilation optimization flags to see how compiler optimization affects 
# the execution time.

# Compile the program with different optimization flags
if [[ $1 == "-c" ]]
then
    sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O0/g' Makefile
    make -j4
    cp -r bin bin0

    sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O1/g' Makefile
    make -j4
    cp -r bin bin1

    sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O2/g' Makefile
    make -j4
    cp -r bin bin2

    sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O3/g' Makefile
    make -j4
    cp -r bin bin3
fi
# make an array to store the execution time of each kernel
declare -a D2
declare -a CA57

#assign big values to the arrays
for i in {1..5}
do
    D2[$i]=100000000
    CA57[$i]=100000000
done

for KERNEL in "default" "default_nb" "optim1" "optim2" "optim3" "optim4" 
do 
    #assign big values to the arrays
    for i in {0..3}
    do
        D2[$i]=100000000
        CA57[$i]=100000000
    done

    for i in {1..5} 
    do
        echo "Running kernel $KERNEL" >&2
        # O0
        D2_0=$(taskset -c 1 ./bin0/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        CA57_0=$(taskset -c 0 ./bin0/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)

        echo "D2_0: $D2_0 ms" >&2
        echo "CA57_0: $CA57_0 ms" >&2

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_0 < ${D2[0]}" | bc) -eq 1 ]
        then
            D2[0]=$D2_0
            echo "update D2_0" >&2
        fi

        if [ $(echo "$CA57_0 < ${CA57[0]}" | bc) -eq 1 ]
        then
            CA57[0]=$CA57_0
            echo "update CA57_0" >&2
        fi

        # O1
        D2_1=$(taskset -c 1 ./bin1/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "D2_1: $D2_1 ms" >&2
        CA57_1=$(taskset -c 0 ./bin1/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "CA57_1: $CA57_1 ms" >&2

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_1 < ${D2[1]}" | bc) -eq 1 ]
        then
            D2[1]=$D2_1
            echo "update D2_1" >&2
        fi

        if [ $(echo "$CA57_1 < ${CA57[1]}" | bc) -eq 1 ]
        then
            CA57[1]=$CA57_1
            echo "update CA57_1" >&2
        fi

        # O2
        D2_2=$(taskset -c 1 ./bin2/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "D2_2: $D2_2 ms" >&2
        CA57_2=$(taskset -c 0 ./bin2/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "CA57_2: $CA57_2 ms" >&2

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_2 < ${D2[2]}" | bc) -eq 1 ]
        then
            D2[2]=$D2_2
            echo "update D2_2" >&2
        fi

        if [ $(echo "$CA57_2 < ${CA57[2]}" | bc) -eq 1 ]
        then
            CA57[2]=$CA57_2
            echo "update CA57_2" >&2
        fi

        ## O3
        D2_3=$(taskset -c 1 ./bin3/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "D2_3: $D2_3 ms" >&2
        CA57_3=$(taskset -c 0 ./bin3/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "CA57_3: $CA57_3 ms" >&2

        # if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_3 < ${D2[3]}" | bc) -eq 1 ]
        then
            D2[3]=$D2_3
            echo "update D2_3">&2
        fi

        if [ $(echo "$CA57_3 < ${CA57[3]}" | bc) -eq 1 ]
        then
            CA57[3]=$CA57_3
            echo "update CA57_3" >&2
        fi
    done

    # print the best execution time for each kernel
    echo "-----------------------------------"
    echo "----------KERNEL $KERNEL-----------"
    echo "-----------------------------------"
    echo 
    echo "#############  O0  ###############"
    echo "D2: ${D2[0]} ms"
    echo "CA57: ${CA57[0]} ms"
    echo "#############  O1  ###############"
    echo "D2: ${D2[1]} ms"
    echo "CA57: ${CA57[1]} ms"
    echo "#############  O2  ###############"
    echo "D2: ${D2[2]} ms"
    echo "CA57: ${CA57[2]} ms"
    echo "#############  O3  ###############"
    echo "D2: ${D2[3]} ms"
    echo "CA57: ${CA57[3]} ms"
done