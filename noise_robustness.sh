#!/bin/bash

for seed in 5 6 7 8 9 10
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=7 python3 memory_retrieval_noise.py --noise_level $noise_level --activation $act --data $data --mode "Man" --seed $seed

            done
        done
    done
done

for seed in 5 6 7 8 9 10
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0    
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=7 python3 memory_retrieval_noise.py --noise_level $noise_level --activation $act --data $data --mode "L2" --seed $seed

            done
        done
    done
done



for seed in 5 6 7 8 9 10
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0
    do
        for act in "softmax" "sparsemax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=7 python3 memory_retrieval_noise.py --noise_level $noise_level --activation $act --data $data --mode "MHN" --seed $seed

            done
        done
    done
done

for seed in 5 6 7 8 9 10
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0
    do
        for act in "poly-10"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=7 python3 memory_retrieval_noise.py --noise_level $noise_level --activation $act --data $data --mode "MHN" --seed $seed

            done
        done
    done
done


for seed in 1 2 3 4 5 6 7 8 9 10
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0
    do
        for kernel_epoch in 100;
        do
            for act in "softmax" "sparsemax"
            do
                for data in "mnist" "cifar10"
                do
                    CUDA_VISIBLE_DEVICES=7 python3 memory_retrieval_noise.py --noise_level $noise_level --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed

                done
            done
        done
    done
done

# for seed in 1 2 3 4
# do
#     for noise_level in 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0 3.0 4.0
#     do
#         for kernel_epoch in 100;
#         do

#             for act in "softmax" "sparsemax"
#             do
#                 for data in "mnist" "cifar10"
#                 do
#                     CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 python3 memory_retrieval.py --noise_level $noise_level --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed

#                 done
#             done
#         done
#     done
# done


