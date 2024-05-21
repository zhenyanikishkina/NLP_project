python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/msmarco-data \
    --weights 100 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 200000 \
    --val_check_interval 10000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --gpus 3


python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/synthetic \
    --weights 100 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --checkpoint_path /home/matfu21/experiments/iterative/multi-obj-repr-learning-master/logs/lightning_logs/version_13/checkpoints/epoch=2_step=62982_avg_val_loss=0.817.ckpt \
    --gpus 1


python evaluate.py \
    --models /home/matfu21/experiments/iterative/multi-obj-repr-learning-master/logs/lightning_logs/version_15/checkpoints/epoch=2_step=97132_avg_val_loss=0.058.ckpt\
    --datasets scidocs scifact arguana \
    --batch_size 32

python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/independent-cropping \
    --weights 100 \
    --batch_size 1 \
    --num_workers 16 \
    --steps 1 \
    --val_check_interval 1 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --gpus 0

python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/unarxiv-d2d ./datasets/training-data/unarxiv-q2d ./datasets/training-data/msmarco-data \
    --weights 33 33 34 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --checkpoint_path /home/matfu21/experiments/iterative/multi-obj-repr-learning-master/logs/lightning_logs/version_3/checkpoints/epoch=6_step=60620_avg_val_loss=0.310.ckpt\
    --gpus 0

python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/independent-cropping ./datasets/training-data/unarxiv-d2d ./datasets/training-data/unarxiv-q2d ./datasets/training-data/msmarco-data \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling alternate \
    --fp16 \
    --gpus 1

python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/unarxiv-d2d ./datasets/training-data/unarxiv-q2d ./datasets/training-data/msmarco-data \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling alternate \
    --fp16 \
    --checkpoint_path /home/matfu21/experiments/iterative/multi-obj-repr-learning-master/logs/lightning_logs/version_3/checkpoints/epoch=6_step=60620_avg_val_loss=0.310.ckpt\
    --gpus 3

----
version_3 - IC
version_4 - q2d
version_5 - d2d
version_6 - msmarco
version_7 - IC + msmarco
version_8 - IC + q2d
version_9 - IC + d2d
version_10 - multi target in-batch mixing
version_11 - multi target alt mixing
version_12 - IC + multi target in-batch mixing
version_13 - IC + multi target alt mixing
version_14 - msmarco + inpars
version_15 - alt mixing + inpars