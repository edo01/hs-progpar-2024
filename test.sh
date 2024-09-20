#!/bin/bash

taskset -c 1 ./bin/easypap -k blur --load-image images/1024.png -i 5 -v seq -wt default_nb -n -sh 1>&2 2>/dev/null 

for KERNEL in "optim1" "optim2" "optim3" "optim4"
do
    echo "test $KERNEL"
    taskset -c 1 ./bin/easypap -k blur --load-image images/1024.png -i 5 -v seq -wt $KERNEL -n -sh 1>&2 2>/dev/null
    diff data/hash/blur-seq-$KERNEL-dim-1024-iter-5-arg-none.sha256 data/hash/blur-seq-default_nb-dim-1024-iter-5-arg-none.sha256 && echo "OK" || echo "TEST $KERNEL FAILED"
done