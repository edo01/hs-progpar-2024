#!/bin/bash
taskset -c 1 ./bin/easypap -k blur --load-image images/1024.png -i 5 -v seq -wt default -n -sh 1>&2 2>/dev/null 

taskset -c 1 ./bin/easypap -k blur --load-image images/1024.png -i 5 -v seq -wt default_nb -n -sh 1>&2 2>/dev/null 
taskset -c 1 ./bin/easypap -k blur_v2 --load-image images/1024.png -i 5 -v seq -wt default_nb -n -sh 1>&2 2>/dev/null 

for V in "" "_v2"
do
    for KERNEL in "optim1" "optim2" "optim3" "optim4"
    do
        echo "test $KERNEL$V"
        taskset -c 1 ./bin/easypap -k blur$V --load-image images/1024.png -i 5 -v seq -wt $KERNEL -n -sh 1>&2 2>/dev/null
        diff data/hash/blur$V-seq-$KERNEL-dim-1024-iter-5-arg-none.sha256 data/hash/blur$V-seq-default_nb-dim-1024-iter-5-arg-none.sha256 && echo "OK" || echo "TEST $KERNEL FAILED"
    done
    echo "test optim5$V"
    taskset -c 1 ./bin/easypap -k blur$V --load-image images/1024.png -i 5 -v seq -wt optim5 -n -sh 1>&2 2>/dev/null
    diff data/hash/blur$V-seq-optim5-dim-1024-iter-5-arg-none.sha256 data/hash/blur-seq-default-dim-1024-iter-5-arg-none.sha256 && echo "OK" || echo "TEST optim5 FAILED"
done
