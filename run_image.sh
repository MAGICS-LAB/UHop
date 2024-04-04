
for datasize in 1000 5000 10000 20000 40000 60000 80000 95000 
do  
    python3 deep_ViH.py --data tiny_imagenet --datasize $datasize --n_class 200 --init_lr 0.0001
done

for datasize in 1000 2000 5000 10000 20000 50000
do  
    python3 image_classification.py --data cifar10 --datasize $datasize     CUDA_VSIBLE_DEVICES=2 python3 deep_ViH.py --data tiny_imagenet --datasize 60000 --n_class 200 --init_lr 0.0001 --batch_size 1024

done

for datasize in 1000 2000 5000 10000 20000 50000
do  
    python3 run_image_bilevel.py --data cifar10 --datasize $datasize
done

for datasize in 1000 5000 10000 20000 40000 60000 80000 95000 
do  
    python3 run_image_bilevel.py --data tiny_imagenet --datasize $datasize --n_class 200
done

