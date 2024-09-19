#!/bin/bash

# for each kernel optimization, we run the program multiple times to get the best execution time. 
# In this way we can obtain a more accurate result, since the execution time can vary depending on the system load.

# Then we run the program using different compilation optimization flags to see how compiler optimization affects 
# the execution time.

# Compile the program with different optimization flags

#sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O0/g' Makefile
#make -j4
#cp -r bin bin0
#
#sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O1/g' Makefile
#make -j4
#cp -r bin bin1
#
#sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O2/g' Makefile
#make -j4
#cp -r bin bin2
#
#sed -ie 's/^CFLAGS\s\+:= -O[0-9]/CFLAGS := -O3/g' Makefile
#make -j4
#cp -r bin bin3

# make an array to store the execution time of each kernel
declare -a D2
declare -a CA52

#assign big values to the arrays
for i in {1..5}
do
    D2[$i]=100000000
    CA52[$i]=100000000
done

for KERNEL in "deafult" "default_nb" "optim1" "optim2" "optim3"
do 
    #assign big values to the arrays
    for i in {1..3}
    do
        D2[$i]=100000000
        CA52[$i]=100000000
    done

    for i in {1..5} 
    do
        # O0

        D2_0=$(taskset -c 0 ./bin0/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        CA52_0=$(taskset -c 1 ./bin0/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)

        echo "D2_0: $D2_0 ms"
        echo "CA52_0: $CA52_0 ms"

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_0 < ${D2[0]}" | bc) -eq 1 ]
        then
            D2[0]=$D2_0
            echo "update D2_0"
        fi

        if [ $(echo "$CA52_0 < ${CA52[0]}" | bc) -eq 1 ]
        then
            CA52[0]=$CA52_0
            echo "update CA52_0"
        fi

        # O1
        D2_1=$(taskset -c 0 ./bin1/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        CA52_1=$(taskset -c 1 ./bin1/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_1 < ${D2[1]}" | bc) -eq 1 ]
        then
            D2[1]=$D2_1
            echo "update D2_1"
        fi

        if [ $(echo "$CA52_1 < ${CA52[1]}" | bc) -eq 1 ]
        then
            CA52[1]=$CA52_1
            echo "update CA52_1"
        fi

        # O2
        D2_2=$(taskset -c 0 ./bin2/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        CA52_2=$(taskset -c 1 ./bin2/easypap -k blur --no-display --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)

        ## if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_2 < ${D2[2]}" | bc) -eq 1 ]
        then
            D2[2]=$D2_2
            echo "update D2_2"
        fi

        if [ $(echo "$CA52_2 < ${CA52[2]}" | bc) -eq 1 ]
        then
            CA52[2]=$CA52_2
            echo "update CA52_2"
        fi

        ## O3
        D2_3=$(taskset -c 0 ./bin3/easypap -k blur --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "D2_3: $D2_3 ms"
        CA52_3=$(taskset -c 1 ./bin3/easypap -k blur --load-image images/1024.png -i 100 -v seq -wt $KERNEL 2>&1| tail -1)
        echo "CA52_3: $CA52_3 ms"

        # if the execution time is less than the previous one, we update the value in the array
        if [ $(echo "$D2_3 < ${D2[3]}" | bc) -eq 1 ]
        then
            D2[3]=$D2_3
            echo "update D2_3"
        fi

        if [ $(echo "$CA52_3 < ${CA52[3]}" | bc) -eq 1 ]
        then
            CA52[3]=$CA52_3
            echo "update CA52_3"
        fi
    done

    # print the best execution time for each kernel
    echo "-----------------------------------"
    echo "----------KERNEL $KERNEL-----------"
    echo "-----------------------------------"
    echo 
    echo "#############  O0  ###############"
    echo "D2: ${D2[0]} ms"
    echo "CA52: ${CA52[0]} ms"
    echo "#############  O1  ###############"
    echo "D2: ${D2[1]} ms"
    echo "CA52: ${CA52[1]} ms"
    echo "#############  O2  ###############"
    echo "D2: ${D2[2]} ms"
    echo "CA52: ${CA52[2]} ms"
    echo "#############  O3  ###############"
    echo "D2: ${D2[3]} ms"
    echo "CA52: ${CA52[3]} ms"
done