import argparse
import os
import numpy as np
from typing import Optional


def process_log(log_dir: Optional[str] = None ,exp_name: Optional[str] = None):
    if log_dir == None or exp_name == None:
        raise ValueError("No log file or path is specified")
    
    micro_f1_dict = dict()
    macro_f1_dict = dict()
    key_list = []

    for filename in os.listdir(log_dir):
        if exp_name in filename:
            file_path = os.path.join(log_dir, filename)
            with open(file_path, 'r') as f:
                data = f.readlines()
                data = data[-4:]

                for line in data:
                    tmp = line.strip().split("|")
                    key = tmp[-2].strip()
                    if key not in key_list:
                        key_list.append(key)
                    f1 = tmp[-1].strip().split(",")
                    micro_f1_dict.setdefault(key, []).append(float(f1[0].strip().split(':')[1].strip()))
                    macro_f1_dict.setdefault(key, []).append(float(f1[1].strip().split(':')[1].strip()))

    print("final result: ")
    for client_id in key_list:
        micro_f1 = np.array(micro_f1_dict[client_id])
        macro_f1 = np.array(macro_f1_dict[client_id])
        print(f'{client_id} | micro_f1: {micro_f1.mean():.4f} ± {micro_f1.std():.4f}, macro_f1: {macro_f1.mean():.4f} ± {macro_f1.std():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # log name
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--task_name', default=None)
    parser.add_argument('--method', default=None)

    args = parser.parse_args()

    exp_name = args.task_name + "_" + args.method

    process_log(args.log_dir, exp_name)