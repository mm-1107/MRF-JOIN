import numpy as np
import pandas as pd
import networkx as nx
from PrivMRF.markov_random_field import MarkovRandomField
from PrivMRF.utils.tools import read_csv, write_csv, generate_column_data, dp_entropy
from PrivMRF.preprocess import preprocess
from exp.evaluate import k_way_marginal
from PrivMRF.domain import Domain
import json
import itertools
import cupy as cp

def mean_noisy_data_num(models):
    mean_noisy_data_num = 0
    for mrf in models:
        mean_noisy_data_num += mrf.noisy_data_num
    return round(mean_noisy_data_num / len(models))


def update_factor_by_data_num(clique_factor, mean_num):
    before  = clique_factor.sum()
    print("before update_factor_by_data_num", before)
    clique_factor.values = clique_factor.values + (mean_num - before) * (clique_factor.values / before)
    print("after update_factor_by_data_num", clique_factor.sum())
    return clique_factor


def update_factor(clique_factor, mean_prob, target):
    xp = cp if clique_factor.xp == "cp"  else np
    before = clique_factor.project(target).values
    attrs = list(set(clique_factor.domain.attr_list) - set([target]))
    domain_remains = clique_factor.project(attrs).domain
    domain_size = domain_remains.size()
    print(f"#DEBUG before clique_factor.project({attrs[0]}).values {clique_factor.project(attrs[0]).values}")
    # index of domain list
    index_list = clique_factor.domain.index_list(attrs)
    target_index = clique_factor.domain.index_list([target])
    # print(f"size of domain_remains = {domain_size}, remain index_list of {attrs} = {index_list}")
    # print(f"index_list of {target} = {target_index}")
    # print("before clique_factor", clique_factor.values)
    print("mean_prob", mean_prob, "before", before, "diff", (mean_prob - before))
    def func(marginal):
        marginal = marginal + (mean_prob - before) / domain_size
        return marginal
    # clique_factor.values = clique_factor.values + (mean_prob - before) / domain_size
    clique_factor.values = xp.apply_along_axis(func1d=func, axis=target_index[0],
                                               arr=clique_factor.values)

    while xp.sum(clique_factor.values < 0) > 0:
        neg_sum = xp.zeros_like(before)
        num_neighbor = xp.zeros_like(before)
        def sum_neg(marginal,neg_sum,num_neighbor):
            num_neighbor += xp.where(marginal > 0, 1, 0)
            neg_sum += xp.where(marginal < 1e-5, marginal, 0)
            return marginal
        xp.apply_along_axis(func1d=sum_neg,
                             axis=target_index[0],
                            arr=clique_factor.values,
                            neg_sum=neg_sum,
                            num_neighbor=num_neighbor)
        print("neg_sum =", neg_sum)
        subtract =  xp.zeros_like(before)
        for domain_idx in range(neg_sum.shape[0]):
            subtract[domain_idx] = neg_sum[domain_idx] / num_neighbor[domain_idx]

        def neg(marginal):
            marginal = xp.where(marginal <= 0, 0, marginal + subtract)
            return marginal
        clique_factor.values = xp.apply_along_axis(func1d=neg, axis=target_index[0],
                                                    arr=clique_factor.values)
    print(f"#DEBUG after clique_factor.project({attrs[0]}).values {clique_factor.project(attrs[0]).values}")
    print("After num_negative", xp.sum(clique_factor.values < 0))
    return clique_factor


def pandas_generate_cond_column_data(df, noisy_data_num, clique_factor, cond, target, all_attr_list):
    clique_factor = clique_factor.moveaxis(all_attr_list)

    if len(cond) == 0:
        # P[target]
        prob = clique_factor.project(target).values
        df.loc[:, target] = generate_column_data(prob, noisy_data_num)
        print(f"marginal_value {target}={prob}")
    else:
        # hist[target, cond_attr1, cond_attr2,...]
        marginal_value = clique_factor.project(cond + [target])

        attr_list = marginal_value.domain.attr_list.copy()
        attr_list.remove(target)
        cond = attr_list.copy()
        attr_list.append(target)

        marginal_value = marginal_value.moveaxis(attr_list).values
        # if len(cond) <= 2:
        #     print(f"marginal_value {attr_list}={marginal_value}")
        def foo(group):
            idx = group.name
            vals = generate_column_data(marginal_value[idx], group.shape[0])
            # print(idx, target)
            # if map_attrB:
            #     target = map_attrB[target]
            group[target] = vals
            return group
        # print("[DEBUG]", self.df.columns)
        df.reset_index(drop = True, inplace = True)
        df = df.groupby(list(cond)).apply(foo)
    return df


def synthetic(models, consistency):
    share_attr = []
    all_attr_list = []
    for idx, mrf in enumerate(models):
        attr_party = mrf.domain.attr_list
        print(f"attr = {attr_party}")
        if idx == 0:
            share_attr = attr_party
        elif idx == 1:
            share_attr = list(set(share_attr) & set(attr_party))
        else:
            share_attr += list(set(all_attr_list) & set(attr_party))
        print("share_attr", share_attr)
        all_attr_list += attr_party

    all_attr_list = list(set(all_attr_list))
    print("All attribute: ", all_attr_list)
    print("share_attr: ", share_attr)
    # all_attr_list = list(set(attrA+attrB))
    if consistency:
        noisy_data_num = mean_noisy_data_num(models)
    else:
        noisy_data_num = models[0].noisy_data_num
    data = np.zeros((noisy_data_num, len(all_attr_list)), dtype=int)
    df = pd.DataFrame(data, columns=all_attr_list)
    # belief propagation to get clique marginals and
    # generate data conditioned on separators
    clique_marginal_A, partition_func = models[0].belief_propagation(models[0].potential)
    prob_for_consistency = dict()
    cnt_for_consistency = dict()
    for mrf in models:
        clique_marginal, partition_func = mrf.belief_propagation(mrf.potential)
        for clique in clique_marginal:
            print(clique, prob_for_consistency.keys())
            # new_clique = tuple([attr+len(attrA)-1 for attr in clique])
            clique_marginal_A[clique] = clique_marginal[clique]
            if consistency:
                clique_marginal_A[clique] = update_factor_by_data_num(clique_marginal_A[clique], noisy_data_num)
                targets = set(share_attr) & set(clique)
                if len(targets) > 0:
                    print(targets, share_attr, clique)
                    for target in targets:
                        if target in prob_for_consistency:
                            cnt_for_consistency[target] += 1
                            prob_for_consistency[target] += clique_marginal[clique].project(target).values
                        else:
                            cnt_for_consistency[target] = 1
                            prob_for_consistency[target] = clique_marginal[clique].project(target).values
                        print("target, cnt", prob_for_consistency[target], cnt_for_consistency[target])
        # clique_marginal_B[new_clique] = tmp_clique_marginal_B[clique]
        # clique_marginal_B.pop(clique)
    clique_marginal_list = list(clique_marginal_A.keys())
    for target in prob_for_consistency:
        # Mean of shared attr
        mean_prob = prob_for_consistency[target] / cnt_for_consistency[target]
        #mean_prob = mean_prob + (noisy_data_num - mean_prob.sum()) * (mean_prob / mean_prob.sum())
        #df.loc[:, target] = generate_column_data(mean_prob, noisy_data_num)
        #print(f"# DEBUG Sum of 0 in {target} = {(df[target] == 0).sum()}")
        for clique in clique_marginal_list:
            if target in clique:
                print("if target in clique:", clique)
                #clique_marginal_A[clique] = update_factor_by_data_num(clique_marginal_A[clique], noisy_data_num)
                clique_marginal_A[clique] = update_factor(clique_marginal_A[clique], mean_prob, target)
                print("#DEBUG  clique_marginal_A[clique].project(target).values",  clique_marginal_A[clique].project(target).values)
    finished_attr = set()
    separator = set()
    for idx, start in enumerate(clique_marginal_list):
        clique = clique_marginal_list[idx+1]
        print(f"start = {start}, clique = {clique}")
        if len(finished_attr) == 0:
            cond_attr =  []
            for attr in start:
                # if attr in prob_for_consistency:
                #     finished_attr.add(attr)
                #     continue
                print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))
                df = pandas_generate_cond_column_data(df, noisy_data_num,
                    clique_marginal_A[start], cond_attr, attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)

        separator = set(start) & set(clique)
        print('start: {}, clique: {}, sep: {}'.format(start, clique, separator))
        cond_attr = list(share_attr) if list(separator) == [] else list(separator)
        for attr in clique:
            if attr not in finished_attr:
                print('  cond_attr: {}, attr: {} {}/{}'.format(cond_attr, attr, len(finished_attr), len(all_attr_list)))
                df = pandas_generate_cond_column_data(df, noisy_data_num,
                    clique_marginal_A[clique], cond_attr,
                    attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)
        if idx == len(clique_marginal_A) - 2:
            break

    print(df)
    #print(f"Sum of 0 in {target} = {(df[list(share_attr)[0]] == 0).sum()}")

    data_list = list(df.to_numpy())
    # tools.write_csv(data_list, all_attr_list), path)
    return data_list


def concat(num_party=2, data_name="acs", exp_name="test", epsilon=0.4, consistency=False):
    _, all_attr_list = read_csv('./data/' + data_name + '.csv')

    # model = MarkovRandomField.load_model('./temp/party_' + data_name + '_model.mrf')
    # data_list = model.synthetic_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
    # marginal_exp([data_list], data_name)

    parties = [chr(65+i) for i in range(num_party)] # A,B,...Z
    models = []
    for party in parties:
        print(f"# Distirbution from Party {party} #")
        path = f'./temp/{exp_name}_{data_name}_{epsilon}_party{party}_model.mrf'
        print(path)
        models.append(MarkovRandomField.load_model(path))

    data_list = synthetic(models, consistency)
    write_csv(data_list, all_attr_list, './out/PrivMRF'+'_'+data_name+'_'+exp_name+'.csv')
    return data_list

def marginal_exp(dp_data_list, data_name="acs", marginal_num=300):
    ways = [3,4,5]
    preprocess(data_name, party="")
    tvd_dict = {}
    for k in ways:
        tvd_list = k_way_marginal(data_name, dp_data_list, k, marginal_num)
        print('      {} way marginal {}'.format(k, tvd_list))
        tvd_dict[str(k)] = tvd_list
        # if os.path.exists('./result/'+temp_exp_name+'_log.json'):
        #     with open('./result/'+temp_exp_name+'_log.json', 'r') as in_file:
        #         result = json.load(in_file)
        # else:
        #     result = {}
        # if str(epsilon) not in result:
        #     result[str(epsilon)] = {}
        # if data_name not in result[str(epsilon)]:
        #     result[str(epsilon)][data_name] = {}
        # if str(i) not in result[str(epsilon)][data_name]:
        #     result[str(epsilon)][data_name][str(i)] = {}
        # result[str(epsilon)][data_name][str(i)][str(k)] = tvd_list
    return tvd_dict

def eval_diff_MI(data_name, syn):
    raw, headings = read_csv('./preprocess/' + data_name + '.csv', print_info=False)
    raw = np.array(raw, dtype=int)
    syn = np.array(syn, dtype=int)
    attr_num = raw.shape[1]
    domain = json.load(open('./preprocess/'+data_name+'.json'))
    domain = {int(key): domain[key] for key in domain}
    domain = Domain(domain, list(range(attr_num)))

    diff = 0
    num_pairs = 0
    for pair in itertools.combinations(domain.attr_list, 2):
        attr_A = pair[0]
        attr_B = pair[1]
        raw_MI = -dp_entropy({}, raw, domain, [attr_A, attr_B], 0)[0]
        raw_MI += dp_entropy({}, raw, domain, [attr_A], 0)[0]
        raw_MI += dp_entropy({}, raw, domain, [attr_B], 0)[0]

        syn_MI = -dp_entropy({}, syn, domain, [attr_A, attr_B], 0)[0]
        syn_MI += dp_entropy({}, syn, domain, [attr_A], 0)[0]
        syn_MI += dp_entropy({}, syn, domain, [attr_B], 0)[0]

        diff += abs(raw_MI - syn_MI)
        num_pairs += 1
        print(f"Mutual information of {attr_A} and {attr_B} -> raw: {raw_MI}, syn: {syn_MI}")
    mid = diff/num_pairs
    print("Evaluation of mutual information =", mid)
    return mid

if __name__ == '__main__':
    data_name = "dummy_12_8_100000"
    data_list = concat(num_party=2, data_name=data_name)
    marginal_exp([data_list], data_name)
