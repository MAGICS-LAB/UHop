#!/bin/bash

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 10 20 30 50 100 200 500;
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                python3 memory_retrieval_max_loss.py --memory_size $mem_size --activation $act --data $data --mode "Man" --seed $seed --rerun 10

            done
        done
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 10 20 30 50 100 200 500;
    do
        for act in "softmax"
        do
            for data in "mnist" "cifar10"
            do
                python3 memory_retrieval_max_loss.py --memory_size $mem_size --activation $act --data $data --mode "L2" --seed $seed --rerun 10
            done
        done
    done
done



for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 10 20 30 50 100 200 500;
    do
        for act in "softmax" "sparsemax"
        do
            for data in "mnist" "cifar10"
            do
                python3 memory_retrieval_max_loss.py --memory_size $mem_size --activation $act --data $data --mode "MHN" --seed $seed --rerun 10
            done
        done
    done
done

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 10 20 30 50 100 200 500;
    do
        for act in "poly-10"
        do
            for data in "mnist" "cifar10"
            do
                python3 memory_retrieval_max_loss.py --memory_size $mem_size --activation $act --data $data --mode "MHN" --seed $seed --rerun 10
            done
        done
    done
done


for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 10 20 30 50 100 200 500;
    do
        for kernel_epoch in 1 2 5 10 50 100 200 500 1000;
        do
            for act in "softmax" "sparsemax"
            do
                for data in "mnist" "cifar10"
                do
                    python3 memory_retrieval_max_loss.py --memory_size $mem_size --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed --rerun 11
                done
            done
        done
    done
done




for seed in 1 2 3 4 5
do
    for mem_size in 10 20 50 100 200 500;
    do
        for kernel_epoch in 1 2 5 10 20 40 70 100 200;
        do

            for act in "softmax" "sparsemax"
            do
                for data in "mnist" "cifar10"
                do
                    python3 memory_retrieval_max_loss.py --memory_size $mem_size --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed --rerun 1
                done
            done
        done
    done
done

