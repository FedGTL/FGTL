#!/bin/sh
# chmod +x scripts/acdne.sh
# nohup scripts/acdne.sh > scripts/acdne.out 2>&1 &

gpu=cuda:0
method=acdne

task_name=fl_ac2d_3

python code/main.py --data_path data/dataset1/ac2d_3.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 30 \
               --local_epoch 10 \
               --check_per_round 5 \
               --batch_size 128 \
               --dropout 0.5 \
               --net_pro_w 0.1 \
               --domain_coef 1 \
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
               --max_round 30 \
               --local_epoch 10 \
               --check_per_round 5 \
               --batch_size 128 \
               --dropout 0.5 \
               --net_pro_w 0.1 \
               --domain_coef 1 \
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
               --max_round 30 \
               --local_epoch 10 \
               --check_per_round 5 \
               --batch_size 128 \
               --dropout 0.5 \
               --net_pro_w 0.1 \
               --domain_coef 1 \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method