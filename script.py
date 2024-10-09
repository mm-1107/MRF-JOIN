import os
import argparse

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
from time import time

def eps_per_party(epsilon, num_party):
    total_to_party = {
         0.4: {2: 0.270},
         0.8: {2: 0.540, 4: 0.370, 8: 0.250},
         1.6: {2: 1.090},
         3.2: {2: 2.160}
    }
    return total_to_party[epsilon][num_party]


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--task', type=str, default='TVD') #tvd/svm
parser.add_argument('--dataset', type=str, default='nltcs')
parser.add_argument('--party', type=int, default=2)
parser.add_argument('--chain', type=bool, default=False)


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

    # number of experiments
    repeat = 10

    num_party = args.party

    parties = [chr(65+i) for i in range(num_party)] # A,B,...Z
    # headings = {"A": [0, 1, 2, 9],
    #     "B": [3, 4, 5, 9],
    #     "C": [6, 7, 8, 9, 10],
    #     "D": [9, 11, 12, 13, 14]
    #     }
    epsilon = eps_per_party(args.epsilon, num_party)
    common_dict = {"adult": [0], "nltcs": [7], "acs": [12], "br2000": [3]}
    share_attr = common_dict[data_name]

    _, all_headings = tools.read_csv('./data/' + data_name + '.csv')
    attr_num = len(all_headings)
    attr_dist = {
        1: { # shared attr num
            12: {2: [6,5], 4: [3, 3, 3, 2], 8: [2, 2, 2, 1, 1, 1, 1, 1]},
            13: {2: [6,6], 4: [3, 3, 3, 3], 8: [2, 2, 2, 2, 1, 1, 1, 1]},
            14: {2: [7,6], 4: [4, 3, 3, 3], 8: [2, 2, 2, 2, 2, 1, 1, 1]},
            15: {2: [7,7], 4: [4, 4, 3, 3], 8: [2, 2, 2, 2, 2, 2, 1, 1]},
            16: {2: [8,7], 4: [4, 4, 4, 3], 8: [2, 2, 2, 2, 2, 2, 2, 1]},
            23: {2: [12, 11], 4: [6, 6, 6, 5], 8: [3, 3, 3, 3, 3, 3, 3, 2]}
        },
        2: { # shared attr num
            15: {2: [6,7]},
            16: {2: [7,7]},
        },
        3: { # shared attr num
            15: {2: [6,6]},
            16: {2: [6,7]},
        }
    }
    path = f'./result/{data_name}_{num_party}party_{args.task}.json'
    if os.path.exists(path):
        with open(path, 'r') as in_file:
            result = json.load(in_file)
    else:
        result = {}
    if str(args.epsilon) not in result:
        result[str(args.epsilon)] = {}
    if "w/o co" not in result[str(args.epsilon)]:
        result[str(args.epsilon)]["w/o co"] = {}
    if "w/ co" not in result[str(args.epsilon)]:
        result[str(args.epsilon)]["w/ co"] = {}
    result[str(args.epsilon)]["w/o co"]["3"] = 0
    result[str(args.epsilon)]["w/o co"]["4"] = 0
    result[str(args.epsilon)]["w/o co"]["5"] = 0
    result[str(args.epsilon)]["w/o co"]["mid"] = 0
    result[str(args.epsilon)]["w/ co"]["3"] = 0
    result[str(args.epsilon)]["w/ co"]["4"] = 0
    result[str(args.epsilon)]["w/ co"]["5"] = 0
    result[str(args.epsilon)]["w/ co"]["mid"] = 0
    for i in range(repeat):
        exhead = []
        for idx, party in enumerate(parties):
            if num_party > 1:
                # _, _, _, exhead = preprocess(data_list[0], party, num_party, share_attr, headings[party])
                _, _, _, exhead_ = preprocess(data_name, party, num_party,
                                              attr_dist[len(share_attr)][attr_num][num_party][idx],
                                              share_attr, [], exhead)
                exhead += exhead_
                print(exhead)
                if args.chain:
                    share_attr = [exhead_[-1]]
            else:
                preprocess(data_name)
            if args.task == 'TVD':
                client_start = time()
                run_experiment([data_name], method_list, exp_name, task='TVD',
                               epsilon_list=[epsilon], repeat=1,
                               classifier_num=25, party=party)
                client_end = time()
            else:
                split(data_name)
                run_experiment([data_name], method_list, exp_name, task='SVM',
                               epsilon_list=[epsilon], repeat=1,
                               classifier_num=25, generate=True, party=party)

            print(f"Executed dataset = {data_name}, party = {party}, common = {share_attr}, epsilon = {epsilon}")

        if args.task == 'TVD':
            # server_start = time()
            concat_data = concat.concat(num_party=num_party,
                                        data_name=data_name,
                                        epsilon=epsilon,
                                        consistency=False)
            # server_end = time()
            tmp_dict = concat.marginal_exp([concat_data], data_name)
            mid = concat.eval_diff_MI(data_name=data_name, syn=concat_data)

            # print(tmp_dict)
            result[str(args.epsilon)]["w/o co"]["3"] += tmp_dict["3"][0]
            result[str(args.epsilon)]["w/o co"]["4"] += tmp_dict["4"][0]
            result[str(args.epsilon)]["w/o co"]["5"] += tmp_dict["5"][0]
            result[str(args.epsilon)]["w/o co"]["mid"] += mid

            server_start = time()
            concat_data = concat.concat(num_party=num_party,
                                        data_name=data_name,
                                        epsilon=epsilon,
                                        consistency=True)
            server_end = time()
            tmp_dict = concat.marginal_exp([concat_data], data_name)
            mid = concat.eval_diff_MI(data_name=data_name, syn=concat_data)

            result[str(args.epsilon)]["w/ co"]["3"] += tmp_dict["3"][0]
            result[str(args.epsilon)]["w/ co"]["4"] += tmp_dict["4"][0]
            result[str(args.epsilon)]["w/ co"]["5"] += tmp_dict["5"][0]
            result[str(args.epsilon)]["w/ co"]["mid"] += mid

        else:
            for k in range(5):
                exp_name_ = exp_name+str(epsilon)+'_'+str(k)
                concat.concat(num_party=num_party, data_name=data_name, exp_name=exp_name_,epsilon=epsilon)
            # exp_name = exp_name+str(epsilon_list[0])+'_'+str(k)
            run_experiment([data_name], method_list, exp_name+str(epsilon), task='SVM',
                epsilon_list=[epsilon], repeat=1,
                classifier_num=25, generate=False)

    if args.task == 'TVD':
        with open(path, 'w') as out_file:
            for way in result[str(args.epsilon)]["w/ co"]:
                result[str(args.epsilon)]["w/ co"][way] = result[str(args.epsilon)]["w/ co"][way] / repeat
            for way in result[str(args.epsilon)]["w/o co"]:
                result[str(args.epsilon)]["w/o co"][way] = result[str(args.epsilon)]["w/o co"][way] / repeat

            print(result[str(args.epsilon)])
            print(f"Result is saved in {path}")
            json.dump(result, out_file)
        print(f'\n\nOne client duration = {client_end-client_start}, Server duration = {server_end-server_start}')
