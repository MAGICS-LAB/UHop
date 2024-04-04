#!/bin/bash


for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0;
    do
        for kernel_epoch in 800;
        do
            for act in "softmax" "sparsemax"
            do
                for data in "cifar10" "mnist"
                do
                    CUDA_VISIBLE_DEVICES=3 python3 memory_retrieval_noise_max_loss.py --noise_level $noise_level --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed --rerun 12

                done
            done
        done
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0;
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=3 python3 memory_retrieval_noise_max_loss.py --noise_level $noise_level --activation $act --data $data --mode "Man" --seed $seed --rerun 12

            done
        done
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0;
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=3 python3 memory_retrieval_noise_max_loss.py --noise_level $noise_level --activation $act --data $data --mode "L2" --seed $seed --rerun 12

            done
        done
    done
done



for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0;
    do
        for act in "softmax" "sparsemax"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=3 python3 memory_retrieval_noise_max_loss.py --noise_level $noise_level --activation $act --data $data --mode "MHN" --seed $seed --rerun 12

            done
        done
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
    for noise_level in 0.0 0.01 0.05 0.1 0.3 0.5 0.7 1.0 1.2 1.4 2.0;
    do
        for act in "poly-10"
        do
            for data in "mnist" "cifar10"
            do
                CUDA_VISIBLE_DEVICES=3 python3 memory_retrieval_noise_max_loss.py --noise_level $noise_level --activation $act --data $data --mode "MHN" --seed $seed --rerun 12

            done
        done
    done
done

