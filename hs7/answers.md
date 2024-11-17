/**
##############################################################
##############################################################
##############################################################

AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q

##############################################################
##############################################################
##############################################################
*/

All tests are run on the Jetsons without producing logs, as they have
been already checked before. Both the motion.c and motion_spu.c have 
been run with 100 frames and using buffering.

# TASK 1 (before)
- throughput motion.c: 9 FPS
- throughput motion_spu.c: 9 FPS

Here we have not implemented any pipeline/parallelization of the code so
it is expected that the throughput is the same for both the motion.c and
motion_spu.c.

   -------------------------------------------||------------------------------||--------------------------------
          Statistics for the given task       ||       Basic statistics       ||        Measured latency        
       ('*' = any, '-' = same as previous)    ||          on the task         ||                                
   -------------------------------------------||------------------------------||--------------------------------
-------------   |-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE    |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
                |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------   |-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho    |           compute |       * ||      101 |     2.16 |  21.33 || 21394.09 | 21093.98 | 36726.11
      Morpho    |           compute |       * ||      100 |     2.13 |  21.00 || 21272.97 | 21097.95 | 22853.95
 Sigma_delta    |           compute |       * ||      101 |     1.64 |  16.20 || 16255.53 | 15859.46 | 39616.90
 Sigma_delta    |           compute |       * ||      100 |     1.61 |  15.88 || 16085.28 | 15889.54 | 19516.83
         CCL    |             apply |       * ||      101 |     0.62 |   6.13 ||  6149.90 |  5006.18 |  8447.97
         CCL    |             apply |       * ||      100 |     0.62 |   6.08 ||  6156.57 |  4999.04 |  8326.94
Features_CCA    |           extract |       * ||      101 |     0.44 |   4.34 ||  4351.62 |  2861.79 |  9779.10
Features_CCA    |           extract |       * ||      100 |     0.43 |   4.26 ||  4318.09 |  2852.77 | 10228.10
Features_filter |            filter |       * ||      100 |     0.18 |   1.81 ||  1830.23 |  1168.67 |  4174.69
Features_filter |            filter |       * ||      101 |     0.18 |   1.78 ||  1785.52 |  1192.13 |  3639.87
       Video    |          generate |       * ||      101 |     0.05 |   0.49 ||   493.75 |     0.00 |   592.70
     Delayer    |           produce |       * ||      101 |     0.03 |   0.34 ||   340.01 |   279.55 |   814.53
     Delayer    |          memorize |       * ||      100 |     0.03 |   0.31 ||   313.67 |   260.26 |   372.96
         KNN    |             match |       * ||      100 |     0.00 |   0.04 ||    43.83 |     8.22 |   110.43
    Tracking    |           perform |       * ||      100 |     0.00 |   0.02 ||    17.32 |     6.56 |    55.33
-------------   |-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL    |                 * |       * ||      100 |    10.13 | 100.00 || 1.01e+05 | 93039.03 | 1.66e+05

# Pipeline Parallelism
## Task 1.1

new Throughput: **13 FPS**

Pipeline stage 1 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_1_to_n_0 |            push_1 |       * ||      100 |     2.72 |  37.26 || 27218.02 |     1.70 | 44933.82
 Sigma_delta |           compute |       * ||      101 |     2.30 |  31.55 || 22819.23 | 17156.38 | 2.51e+05
 Sigma_delta |           compute |       * ||      100 |     2.16 |  29.61 || 21634.74 | 17132.42 | 1.27e+05
       Video |          generate |       * ||      101 |     0.05 |   0.75 ||   543.40 |     0.00 |  1940.29
     Delayer |           produce |       * ||      101 |     0.03 |   0.42 ||   306.35 |   249.89 |  1821.47
     Delayer |          memorize |       * ||      100 |     0.03 |   0.41 ||   298.90 |   252.19 |  1760.58
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     7.31 | 100.00 || 73057.34 | 34966.64 | 4.31e+05
Pipeline stage 2 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho |           compute |       * ||      100 |     2.27 |  29.64 || 22737.40 | 21144.22 | 89653.06
      Morpho |           compute |       * ||      100 |     2.22 |  28.99 || 22240.74 | 21124.22 | 76609.66
         CCL |             apply |       * ||      100 |     0.72 |   9.42 ||  7224.58 |  5728.29 | 32461.67
         CCL |             apply |       * ||      100 |     0.71 |   9.26 ||  7102.85 |  5791.52 | 37013.63
Features_CCA |           extract |       * ||      100 |     0.44 |   5.80 ||  4449.78 |  2367.58 | 30514.37
Features_CCA |           extract |       * ||      100 |     0.44 |   5.72 ||  4384.86 |  3238.43 | 30547.75
Adp_1_to_n_0 |            pull_n |       * ||      100 |     0.34 |   4.40 ||  3374.35 |     2.02 | 3.37e+05
Features_filter |            filter |       * ||      100 |     0.26 |   3.41 ||  2616.25 |   652.51 |  9518.98
Features_filter |            filter |       * ||      100 |     0.25 |   3.28 ||  2520.24 |  1558.88 |  4644.16
         KNN |             match |       * ||      100 |     0.01 |   0.09 ||    68.79 |    41.73 |   390.05
Adp_1_to_n_1 |            push_1 |       * ||      100 |     0.00 |   0.00 ||     3.13 |     2.18 |     3.84
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     7.67 | 100.00 || 76722.97 | 61651.59 | 6.49e+05
Pipeline stage 3 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_1_to_n_1 |            pull_n |       * ||      100 |     7.67 |  99.96 || 76679.45 | 64769.28 | 4.71e+05
    Tracking |           perform |       * ||      100 |     0.00 |   0.04 ||    28.39 |     9.15 |    95.68
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     7.67 | 100.00 || 76707.84 | 64778.43 | 4.71e+05


We can observe that the pipeline is limited by the slowest stage, which is the second one. This can be
seen by the fact that the push_1 task of the Adp_1_to_n_0 module of the first stage and the pull_n task of the 
Adp_1_to_n_1 module of the third stage have a critical percentage of 37.26% and 99.96% respectively. This means
that the second stage is the bottleneck of the pipeline. Is not sufficient to increase the buffer size.

## TASK REPLICATION

New Throughput using 2 Threads: **16 FPS**
New Throughput using 4 Threads: **23 FPS**


Pipeline stage 1 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
 Sigma_delta |           compute |       * ||      101 |     2.46 |  42.53 || 24315.22 | 16178.91 | 1.55e+05
 Sigma_delta |           compute |       * ||      100 |     2.31 |  40.09 || 23146.64 | 16190.53 | 1.17e+05
Adp_1_to_n_0 |            push_1 |       * ||      100 |     0.87 |  15.05 ||  8688.02 |     2.40 | 1.52e+05
       Video |          generate |       * ||      101 |     0.06 |   0.97 ||   554.90 |     0.00 |  1272.61
     Delayer |           produce |       * ||      101 |     0.04 |   0.71 ||   405.43 |   255.10 |  1910.98
     Delayer |          memorize |       * ||      100 |     0.04 |   0.66 ||   378.73 |   254.43 |  1139.10
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     5.77 | 100.00 || 57741.70 | 33045.72 | 4.30e+05
Pipeline stage 2 (2 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho |           compute |       * ||      100 |     3.69 |  30.12 || 36875.96 | 22991.36 | 91553.54
      Morpho |           compute |       * ||      100 |     3.56 |  29.06 || 35571.13 | 22993.89 | 89589.54
Adp_1_to_n_0 |            pull_n |       * ||      100 |     1.03 |   8.39 || 10264.68 |     2.11 | 4.28e+05
         CCL |             apply |       * ||      100 |     0.83 |   6.77 ||  8284.86 |  5899.36 | 15838.21
         CCL |             apply |       * ||      100 |     0.82 |   6.74 ||  8247.41 |  5863.65 | 14627.10
Adp_n_to_1_1 |            push_n |       * ||      100 |     0.72 |   5.91 ||  7234.92 |     1.70 | 1.42e+05
Features_CCA |           extract |       * ||      100 |     0.54 |   4.39 ||  5373.73 |  3822.43 | 19880.51
Features_CCA |           extract |       * ||      100 |     0.52 |   4.25 ||  5199.67 |  3832.00 | 10381.12
Features_filter |            filter |       * ||      100 |     0.26 |   2.16 ||  2643.24 |  1575.87 |  6942.21
Features_filter |            filter |       * ||      100 |     0.26 |   2.15 ||  2631.13 |  1542.14 |  8410.08
         KNN |             match |       * ||      100 |     0.01 |   0.07 ||    88.16 |    41.82 |   722.62
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |    12.24 | 100.00 || 1.22e+05 | 68566.34 | 8.29e+05
Pipeline stage 3 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_n_to_1_1 |            pull_1 |       * ||      100 |     6.18 |  99.97 || 61804.49 |     1.38 | 4.66e+05
    Tracking |           perform |       * ||      100 |     0.00 |   0.03 ||    17.53 |     2.85 |    70.27
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     6.18 | 100.00 || 61822.02 |     4.22 | 4.66e+05

We can see that by increasing the number of threads in the pipeline in the second stage, we can 
increase the throughput of the pipeline. This is because the second stage is the bottleneck of the pipeline. 
The effect of increasing the number of threads can be seen by the fact that the percentage of the push_1 task of the
Adp_1_to_n_0 module of the first stage is reduced from 37.26% to 15.05% and the percentage, and even more when 
using 4 threads.

# GRAPH SEMPLIFICATION

Pipeline stage 1 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
 Sigma_delta |           compute |       * ||      100 |     2.02 |  97.24 || 20244.66 | 16712.74 | 1.23e+05
       Video |          generate |       * ||      101 |     0.06 |   2.75 ||   565.89 |     0.00 |  2073.86
Adp_1_to_n_0 |            push_1 |       * ||      100 |     0.00 |   0.01 ||     2.94 |     2.05 |     5.95
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     2.08 | 100.00 || 20819.14 | 16714.79 | 1.25e+05
Pipeline stage 2 (4 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho |           compute |       * ||      100 |     5.03 |  58.64 || 50337.66 | 21255.39 | 2.76e+05
Adp_1_to_n_0 |            pull_n |       * ||      100 |     1.90 |  22.17 || 19031.47 |     2.05 | 1.56e+05
         CCL |             apply |       * ||      100 |     0.91 |  10.56 ||  9063.24 |  5707.10 | 18551.23
Features_CCA |           extract |       * ||      100 |     0.49 |   5.68 ||  4877.03 |  3097.18 | 13512.45
Features_filter |            filter |       * ||      100 |     0.25 |   2.95 ||  2532.31 |  1456.00 |  9022.30
Adp_n_to_1_1 |            push_n |       * ||      100 |     0.00 |   0.00 ||     3.18 |     2.18 |     4.61
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     8.58 | 100.00 || 85844.88 | 31519.91 | 4.73e+05
Pipeline stage 3 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_n_to_1_1 |            pull_1 |       * ||      100 |     2.15 |  98.69 || 21476.32 |     1.22 | 3.19e+05
     Delayer |           produce |       * ||      100 |     0.01 |   0.59 ||   129.07 |    42.02 |   874.69
     Delayer |          memorize |       * ||      100 |     0.01 |   0.33 ||    71.29 |    25.50 |   134.59
         KNN |             match |       * ||      100 |     0.01 |   0.32 ||    70.29 |    50.56 |   216.29
    Tracking |           perform |       * ||      100 |     0.00 |   0.06 ||    12.95 |     4.67 |    81.70
     Delayer |           produce |       * ||      100 |     0.00 |   0.01 ||     1.51 |     0.26 |     3.30
     Delayer |          memorize |       * ||      100 |     0.00 |   0.00 ||     0.57 |     0.26 |     1.82
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     2.18 | 100.00 || 21761.99 |   124.48 | 3.20e+05



*task 1* - New Throughput: **45 FPS** (using 4 threads in stage 2)
By simplifying the graph we can see that the throughput of the pipeline is doubled. This is because we have 
halved the number of tasks in the second stage of the pipeline, that is the most critical one. In fact the 
average latency of the second stage is reduced from 122000 us to 85844 us.

*task 2* - The k-NN task could have been replicated and a 4-stage pipeline could have been implemented, since the k-NN task
does not carry any data dependecy over time. However, the k-NN task takes only 0.01% of the third stage time, so it 
is not worth replicating it. We can suppose that, if we replicate it and we add a new stage, the application will
not benefit from it given the additional overhead of handling a new stage. In fact, replication can be useful 
when the task is computationally expensive and it is blocking the pipeline.

*task 3* - When pinning the threads to the cores, we experienced a slight increase in the throughput of the pipeline
around(3-5 FPS).

# TASK 4 Data Parallelism with OpenMP

Pipeline stage 1 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
 Sigma_delta |           compute |       * ||      100 |     1.53 |  95.35 || 15253.43 |  7469.44 | 26120.87
       Video |          generate |       * ||      101 |     0.07 |   4.25 ||   673.26 |     0.00 |  1114.91
Adp_1_to_n_0 |            push_1 |       * ||      100 |     0.01 |   0.40 ||    63.37 |     1.95 |  5660.35
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     1.60 | 100.00 || 15996.79 |  7471.39 | 32907.28
Pipeline stage 2 (4 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho |           compute |       * ||      100 |     3.91 |  59.32 || 39095.08 | 21127.30 | 56404.96
Adp_1_to_n_0 |            pull_n |       * ||      100 |     1.18 |  17.93 || 11818.50 |     2.08 | 53477.54
         CCL |             apply |       * ||      100 |     0.84 |  12.78 ||  8419.57 |  5952.54 | 10727.84
Features_CCA |           extract |       * ||      100 |     0.41 |   6.22 ||  4099.53 |  3066.69 |  5932.29
Features_filter |           filterf |       * ||      100 |     0.22 |   3.34 ||  2199.16 |  1035.78 |  5277.66
Adp_n_to_1_1 |            push_n |       * ||      100 |     0.03 |   0.41 ||   270.26 |     2.14 | 13074.59
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     6.59 | 100.00 || 65902.10 | 31186.53 | 1.45e+05
Pipeline stage 3 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_n_to_1_1 |            pull_1 |       * ||      100 |     1.64 |  98.54 || 16429.47 |     1.28 | 82083.49
     Delayer |           produce |       * ||      100 |     0.01 |   0.68 ||   113.74 |    58.98 |   220.70
     Delayer |          memorize |       * ||      100 |     0.01 |   0.48 ||    80.59 |    26.78 |   650.11
         KNN |            matchf |       * ||      100 |     0.00 |   0.17 ||    29.06 |     7.78 |   146.50
    Tracking |           perform |       * ||      100 |     0.00 |   0.11 ||    17.80 |     4.61 |    82.56
     Delayer |           produce |       * ||      100 |     0.00 |   0.01 ||     1.50 |     0.32 |     2.91
     Delayer |          memorize |       * ||      100 |     0.00 |   0.00 ||     0.61 |     0.22 |     1.15
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     1.67 | 100.00 || 16672.77 |    99.97 | 83187.43
End of the program, exiting.


*task 1* - New Throughput: **59 FPS** (using 4 threads in stage 2 and 2 OMP threads in the sigma_delta task)
Sigma_delta cannot be replicated because it is a task that has a data dependency over time. However, we can parallelize
the computation of the sigma_delta task using OpenMP. Using OMP we can decrease the average latency of the sigma_delta
20244 us to 15253 us. 

*task 2* - The best configuration is to use 2 OMP threads in the sigma_delta task, while using 4 threads in the second stage.
Moreover, the scheduling policy of the OMP threads is set to static, since the workload is perfectly balanced among the threads 
so we can avoid the overhead of dynamic scheduling. The chunk size is set to 3, in this way each thread will process 3 lines of the
image avoiding the false sharing problem.

# TASK 5 MIPP

Pipeline stage 1 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
 Sigma_delta |           compute |       * ||      100 |     0.68 |  52.16 ||  6827.76 |  3416.54 |  9442.18
Adp_1_to_n_0 |            push_1 |       * ||      100 |     0.55 |  42.24 ||  5529.75 |     1.73 | 45148.23
       Video |          generate |       * ||      101 |     0.07 |   5.60 ||   725.53 |     0.00 |  1258.78
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     1.31 | 100.00 || 13090.29 |  3418.27 | 55861.77
Pipeline stage 2 (4 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      Morpho |           compute |       * ||      100 |     3.49 |  64.20 || 34874.50 | 21105.41 | 57170.15
         CCL |             apply |       * ||      100 |     0.83 |  15.34 ||  8331.08 |  6531.74 | 12075.07
Features_CCA |           extract |       * ||      100 |     0.46 |   8.40 ||  4562.53 |  3734.40 |  7749.41
Adp_1_to_n_0 |            pull_n |       * ||      100 |     0.24 |   4.38 ||  2379.70 |     1.98 | 22632.64
Adp_n_to_1_1 |            push_n |       * ||      100 |     0.22 |   4.07 ||  2211.83 |     2.24 | 30821.41
Features_filter |           filterf |       * ||      100 |     0.20 |   3.61 ||  1961.63 |   952.13 |  4644.22
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     5.43 | 100.00 || 54321.26 | 32327.91 | 1.35e+05
Pipeline stage 3 (1 thread(s)): 
-------------------------------------------||------------------------------||--------------------------------
       Statistics for the given task       ||       Basic statistics       ||        Measured latency        
    ('*' = any, '-' = same as previous)    ||          on the task         ||                                
-------------------------------------------||------------------------------||--------------------------------
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
      MODULE |              TASK |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM 
             |                   |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) 
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
Adp_n_to_1_1 |            pull_1 |       * ||      100 |     1.35 |  98.15 || 13470.31 |     1.12 | 79724.13
     Delayer |           produce |       * ||      100 |     0.01 |   0.77 ||   105.48 |    83.26 |   226.59
     Delayer |          memorize |       * ||      100 |     0.01 |   0.64 ||    87.53 |    26.43 |   465.54
         KNN |            matchf |       * ||      100 |     0.00 |   0.26 ||    35.38 |     9.06 |   114.59
    Tracking |           perform |       * ||      100 |     0.00 |   0.16 ||    22.55 |     4.80 |   105.31
     Delayer |           produce |       * ||      100 |     0.00 |   0.01 ||     1.66 |     0.42 |     3.46
     Delayer |          memorize |       * ||      100 |     0.00 |   0.01 ||     0.78 |     0.26 |     1.44
-------------|-------------------|---------||----------|----------|--------||----------|----------|----------
       TOTAL |                 * |       * ||      100 |     1.37 | 100.00 || 13723.68 |   125.34 | 80641.06

**task 1** - New Throughput: **72 FPS** (using 4 threads in stage 2 and MIPP in the sigma_delta task)
Using MIPP we can further decrease the average latency of the sigma_delta task from 15253 us to 6827 us. 
In this way we are using different parallelization techniques.

**task 2** - With this configuration we can see that now the morpho task is the bottleneck of the pipeline,
taking 34874 us on average. Since we are using already 4 threads in the second stage, we cannot further increase
the throughput of the pipeline by using more threads. This suggests that the configuration is already optimal.
One possible solution could be to further optimize the morpho task by using MIPP.

The total speedup of the pipeline is 72/9 = 8x.

# TASK 6 Data Parallelism versus Pipeline+Replication
Here we simplify the graph also in the base version of the code without pipeline. Moreover we add the data 
parallelization to morpho and sigma_delta tasks without using the vectorization and the pipeline.

Using this configuration we can achieve a throughput of **30 FPS** using static,3 and all threads available.
As we known from computer architectures, the pipeline allows to increase drammatically the throughput 
of a given application by overlapping the execution of different tasks. For this reason, the pipelined
version of the application has a throughput of 72 FPS, which is higher by a factor of 2.4x. 
