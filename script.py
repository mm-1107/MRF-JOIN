import os
import argparse
# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from exp.evaluate import run_experiment, split
import json
import csv
import pandas as pd
import numpy as np

import sys
from PrivMRF.preprocess import preprocess
from PrivMRF.utils import tools
import concat

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--task', type=str, default='TVD') #tvd/svm
parser.add_argument('--dataset', type=str, default='nltcs')


if __name__ == '__main__':
    args = parser.parse_args()
    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)
    # adult, br2000, nltcs, acs
    data_name = args.dataset
    # data_list = ['nltcs', 'acs', 'adult', 'br2000']

    # PrivMRF
    method_list = ['PrivMRF']

    # arbitrary string for naming output data
    exp_name = 'test'

    # 0.1, 0.2, 0.4, 0.8, 1.6, 3.2
    # epsilon_list = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    # epsilon_list = [float('inf')]
    # number of experiments
    repeat = 3

    num_party = 2
    parties = [chr(65+i) for i in range(num_party)] # A,B,...Z
    # headings = {"A": [0, 1, 2, 9],
    #     "B": [3, 4, 5, 9],
    #     "C": [6, 7, 8, 9, 10],
    #     "D": [9, 11, 12, 13, 14]
    #     }
    # sequential composition
    epsilon = args.epsilon/num_party
    common_dict = {"adult": 0, "nltcs": 8, "acs": 12, "br2000": 3}
    common_attr = common_dict[data_name]

    _, all_headings = tools.read_csv('./data/' + data_name + '.csv')
    attr_num = len(all_headings)
    attr_dist = {
        12: {2: [6,5], 4: [3, 3, 3, 2], 8: [2, 2, 2, 1, 1, 1, 1, 1]},
        13: {2: [6,6], 4: [3, 3, 3, 3], 8: [2, 2, 2, 2, 1, 1, 1, 1]},
        14: {2: [7,6], 4: [4, 3, 3, 3], 8: [2, 2, 2, 2, 2, 1, 1, 1]},
        15: {2: [7,7], 4: [4, 4, 3, 3], 8: [2, 2, 2, 2, 2, 2, 1, 1]},
        16: {2: [8,7], 4: [4, 4, 4, 3], 8: [2, 2, 2, 2, 2, 2, 2, 1]},
        23: {2: [12, 11], 4: [6, 6, 6, 5], 8: [3, 3, 3, 3, 3, 3, 3, 2]}
    }
    path = f'./result/{num_party}party_{args.task}.json'
    if os.path.exists(path):
        with open(path, 'r') as in_file:
            result = json.load(in_file)
    else:
        result = {}
    if args.epsilon not in result:
        result[args.epsilon] = {}
    if data_name not in result[args.epsilon]:
        result[args.epsilon][data_name] = {}
    result[args.epsilon][data_name][3] = 0
    result[args.epsilon][data_name][4] = 0
    result[args.epsilon][data_name][5] = 0
    for i in range(repeat):
        exhead = []
        for idx, party in enumerate(parties):
            if num_party > 1:
                # _, _, _, exhead = preprocess(data_list[0], party, num_party, common_attr, headings[party])
                _, _, _, exhead_ = preprocess(data_name, party, num_party,
                                              attr_dist[attr_num][num_party][idx],
                                              common_attr, [], exhead)
                exhead += exhead_
                print(exhead)
            else:
                preprocess(data_name)
            if args.task == 'TVD':
                run_experiment([data_name], method_list, exp_name, task='TVD',
                               epsilon_list=[epsilon], repeat=repeat,
                               classifier_num=25, party=party)
            else:
                split(data_name)
                run_experiment([data_name], method_list, exp_name, task='SVM',
                               epsilon_list=[epsilon], repeat=1,
                               classifier_num=25, generate=True, party=party)

            print(f"Executed dataset = {data_name}, party = {party}, common = {common_attr}, epsilon = {epsilon}")

        if args.task == 'TVD':
            concat_data = concat.concat(num_party=num_party, data_name=data_name)
            tmp_dict = concat.marginal_exp([concat_data], data_name)
            # print(tmp_dict)
            result[args.epsilon][data_name][3] += tmp_dict[str(3)][0]
            result[args.epsilon][data_name][4] += tmp_dict[str(4)][0]
            result[args.epsilon][data_name][5] += tmp_dict[str(5)][0]
        else:
            for k in range(5):
                exp_name = exp_name+str(epsilon)+'_'+str(k)
                concat.concat(num_party=num_party, data_name=data_name, exp_name=exp_name)
            # exp_name = exp_name+str(epsilon_list[0])+'_'+str(k)
            run_experiment([data_name], method_list, exp_name, task='SVM',
                epsilon_list=[epsilon], repeat=1,
                classifier_num=25, generate=False)

    if args.task == 'TVD':
        with open(path, 'w') as out_file:
            result[args.epsilon][data_name][3] = result[args.epsilon][data_name][3] / repeat
            result[args.epsilon][data_name][4] = result[args.epsilon][data_name][4] / repeat
            result[args.epsilon][data_name][5] = result[args.epsilon][data_name][5] / repeat
            print(result)
            json.dump(result, out_file)
