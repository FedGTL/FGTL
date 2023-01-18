#!/bin/sh
# chmod +x scripts/fednova.sh
# nohup scripts/fednova.sh > scripts/fednova.out 2>&1 &

gpu=cuda:0
method=nova

task_name=fl_ac2d_3

python code/main.py --data_path data/dataset1/ac2d_3.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0 \
               --rho 0.9 \
               --batch_size 128 \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method

task_name=fl_ad2c_3

python code/main.py --data_path data/dataset1/ad2c_3.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0 \
               --rho 0.9 \
               --batch_size 128 \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method

task_name=fl_cd2a_3

python code/main.py --data_path data/dataset1/cd2a_3.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0 \
               --rho 0.9 \
               --batch_size 128 \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method