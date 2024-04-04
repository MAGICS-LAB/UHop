# for seed in 1 2 3 4 5 6 7 8 9 10
# do
#     for mem_size in 100;
#     do
#         for kernel_epoch in 1 2 5 10 20 40 70 100 200 250 500 1000 2000;
#         do

#             for act in "softmax" "sparsemax"
#             do
#                 for data in "mnist" "cifar10"
#                 do
#                     CUDA_VISIBLE_DEVICES=5 python3 memory_retrieval_max_loss.py --memory_size $mem_size --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed --rerun 44

#                 done
#             done
#         done
#     done
# done

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for mem_size in 100;
    do
        for kernel_epoch in 1 2 5 10 20 40 70 100 200 250 500 1000 2000;
        do

            for act in "softmax" "sparsemax"
            do
                for data in "mnist" "cifar10"
                do
                    CUDA_VISIBLE_DEVICES=5 python3 memory_retrieval_max_loss.py --memory_size $mem_size --kernel_epoch $kernel_epoch --activation $act --data $data --mode "UMHN" --seed $seed --rerun 44

                done
            done
        done
    done
done

