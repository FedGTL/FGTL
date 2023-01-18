#!/bin/sh
# chmod +x scripts/fgtl+.sh
# nohup scripts/fgtl+.sh > scripts/fgtl+.out 2>&1 &

gpu=cuda:0
method=fgtl

task_name=fl_ac2d_5

python code/main.py --data_path data/dataset1/ac2d_5.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0.5 \
               --warm_up 0 \
               --confidence_gate_begin 0.7 \
               --confidence_gate_end 0.8 \
               --use_lpa True \
               --num_iter 1 \
               --use_ppmi True \
               --path_len 10 \
               --use_contrast True \
               --aug_type edge \
               --drop_percent 0.2 \
               --hop_number 10 \
               --unsuper_round 500 \
               --group True \
               --n_clusters 2 \
               --group_mode hard \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method

task_name=fl_ad2c_5

python code/main.py --data_path data/dataset1/ad2c_5.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0.5 \
               --warm_up 0 \
               --confidence_gate_begin 0.85 \
               --confidence_gate_end 0.9 \
               --use_lpa True \
               --num_iter 1 \
               --use_ppmi True \
               --path_len 10 \
               --use_contrast True \
               --aug_type edge \
               --drop_percent 0.2 \
               --hop_number 10 \
               --unsuper_round 500 \
               --group True \
               --n_clusters 2 \
               --group_mode hard \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method

task_name=fl_cd2a_5

python code/main.py --data_path data/dataset1/cd2a_5.pkl \
               --task_name $task_name \
               --method $method \
               --num_exp 3 \
               --lr 1e-4 \
               --weight_decay 5e-4 \
               --max_round 1500 \
               --local_epoch 10 \
               --check_per_round 5 \
               --dropout 0.5 \
               --warm_up 0 \
               --confidence_gate_begin 0.7 \
               --confidence_gate_end 0.8 \
               --use_lpa True \
               --num_iter 1 \
               --use_ppmi True \
               --path_len 10 \
               --use_contrast True \
               --aug_type edge \
               --drop_percent 0.2 \
               --hop_number 10 \
               --unsuper_round 500 \
               --group True \
               --n_clusters 2 \
               --group_mode hard \
               --num_layers 2 \
               --h_dim 256 \
               --ser_model gcn \
               --cli_model gcn \
               --gpu $gpu

python code/log_statistics.py \
               --log_dir log \
               --task_name $task_name \
               --method $method