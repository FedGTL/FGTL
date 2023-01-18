import argparse
from utils import init_dir
from fedavg import FedAvg
from fedprox import FedProx
from scaffold import Scaffold
from fednova import FedNova
from moon import Moon
from kd3a import KD3A
from acdne import ACDNE
from fgnn import FGNN
from fgtl import FGTL



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and output data
    parser.add_argument('--data_path', default='data/dataset1/ac2d_3.pkl')
    parser.add_argument('--state_dir', default='state')
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--tb_log_dir', default='tb_log')

    # exp name
    parser.add_argument('--task_name', default='fl_ac2d_3')
    parser.add_argument('--method', default='avg', type=str, choices=['avg', 'prox', 'scaffold', 'nova', 'moon', 'acdne', 'kd3a', 'fgnn', 'fgtl'])
    parser.add_argument('--num_exp', default=1, type=int) 

    # exp setting
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--max_round', default=500, type=int)
    parser.add_argument('--local_epoch', default=3, type=int)
    parser.add_argument('--check_per_round', default=1, type=int)

    # method
    parser.add_argument('--mu', default=1e-2, type=float)
    parser.add_argument('--rho', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--warm_up', default=0, type=int, help='the number of warm up round')
    parser.add_argument('--confidence_gate_begin', default=0.8, type=float)
    parser.add_argument('--confidence_gate_end', default=0.95, type=float)
    parser.add_argument('--use_lpa', default=False, type=bool)
    parser.add_argument('--num_iter', default=5, type=int, help='for LPA')
    parser.add_argument('--use_ppmi', default=False, type=bool)
    parser.add_argument('--path_len', default=5, type=int, help='for ppmi')
    parser.add_argument('--use_contrast', default=False, type=bool)
    parser.add_argument('--aug_type', default='node', type=str, choices=['edge', 'node', 'subgraph', 'mask'])
    parser.add_argument('--hop_number', default=10, type=int)
    parser.add_argument('--drop_percent', default=0.2, type=float)
    parser.add_argument('--unsuper_round', default=100, type=int)
    parser.add_argument('--net_pro_w', default=0.1, type=float)
    parser.add_argument('--domain_coef', default=0.1, type=float)
    parser.add_argument('--group', default=False, type=bool)
    parser.add_argument('--n_clusters', default=2, type=int)
    parser.add_argument('--group_mode', default='hard', type=str, choices=['hard', 'soft'])

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--h_dim', default=256, type=int)
    parser.add_argument('--act', default='relu', type=str, choices=['relu', 'sigmoid', 'leakyrelu'])
    parser.add_argument('--ser_model', default='gcn', type=str, choices=['gcn', 'mlp'])
    parser.add_argument('--cli_model', default='gcn', type=str, choices=['gcn', 'mlp'])

    parser.add_argument('--gpu', default='cuda:0', type=str)


    args = parser.parse_args()

    init_dir(args)

    # run exps
    for run in range(args.num_exp):
        args.run = run
        args.exp_name = args.task_name + "_" + args.method + f'_run{args.run}'

        if args.method == 'avg':
            trainer = FedAvg(args)
        elif args.method == 'prox':
            trainer = FedProx(args)
        elif args.method == 'scaffold':
            trainer = Scaffold(args)
        elif args.method == 'nova':
            trainer = FedNova(args)
        elif args.method == 'moon':
            trainer = Moon(args)
        elif args.method == 'kd3a':
            trainer = KD3A(args)
        elif args.method == 'acdne':
            trainer = ACDNE(args)
        elif args.method == 'fgnn':
            trainer = FGNN(args)
        elif args.method == 'fgtl':
            trainer = FGTL(args)

        trainer.train()

        del trainer
    
