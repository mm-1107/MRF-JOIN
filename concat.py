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

def pandas_generate_cond_column_data(df, model, clique_factor, cond, target, all_attr_list):
    clique_factor = clique_factor.moveaxis(all_attr_list)

    if len(cond) == 0:
        # P[target]
        prob = clique_factor.project(target).values
        df.loc[:, target] = generate_column_data(prob, model.noisy_data_num)
    else:
        # P[attr|cond_attr]??
        marginal_value = clique_factor.project(cond + [target])

        attr_list = marginal_value.domain.attr_list.copy()
        attr_list.remove(target)
        cond = attr_list.copy()
        attr_list.append(target)

        marginal_value = marginal_value.moveaxis(attr_list).values

        if model.config['enable_attribute_hierarchy']:
            # TODO
            attrs = [attr for attr in attr_list if attr < model.attr_num and model.attr_to_subattr[attr] in attr_list]
            model.set_zero_for_hierarchy(marginal_value, attr_list, attrs)

        def foo(group):
            idx = group.name
            # この辺を帰る？？
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


def synthetic(models):
    all_attr_list = []
    for mrf in models:
        attr_party = mrf.domain.attr_list
        print(f"attr = {attr_party}")
        all_attr_list += attr_party

    all_attr_list = list(set(all_attr_list))
    print("All attribute: ", all_attr_list)
    # all_attr_list = list(set(attrA+attrB))
    data = np.zeros((models[0].noisy_data_num, len(all_attr_list)), dtype=int)
    df = pd.DataFrame(data, columns=all_attr_list)
    # belief propagation to get clique marginals and
    # generate data conditioned on separators
    clique_marginal_A, partition_func = models[0].belief_propagation(models[0].potential)
    for mrf in models:
        clique_marginal, partition_func = mrf.belief_propagation(mrf.potential)
        for clique in clique_marginal:
            print(clique)
            # new_clique = tuple([attr+len(attrA)-1 for attr in clique])
            # TODO
            clique_marginal_A[clique] = clique_marginal[clique]

        # clique_marginal_B[new_clique] = tmp_clique_marginal_B[clique]
        # clique_marginal_B.pop(clique)
    clique_marginal_list = list(clique_marginal_A.keys())

    finished_attr = set()
    separator = set()
    for idx, start in enumerate(clique_marginal_list):
        clique = clique_marginal_list[idx+1]
        print(f"start = {start}, clique = {clique}")
        if len(finished_attr) == 0:
            cond_attr = []
            for attr in start:
                print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))
                df = pandas_generate_cond_column_data(df, models[0],
                    clique_marginal_A[start], cond_attr, attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)

        separator = set(start) & set(clique)
        print('start: {}, clique: {}, sep: {}'.format(start, clique, separator))
        cond_attr = list(separator)
        for attr in clique:
            if attr not in finished_attr:
                print('  cond_attr: {}, attr: {} {}/{}'.format(cond_attr, attr, len(finished_attr), len(all_attr_list)))
                df = pandas_generate_cond_column_data(df, models[0],
                    clique_marginal_A[clique], cond_attr,
                    attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)
        if idx == len(clique_marginal_A) - 2:
            break

    print(df)
    data_list = list(df.to_numpy())
    # tools.write_csv(data_list, all_attr_list), path)
    return data_list

def synthetic_party(df, model, clique_marginal, all_attr_list, common_attr=[]):
    clique_marginal_list = list(clique_marginal.keys())
    finished_attr = set()
    separator = set()
    for idx, start in enumerate(clique_marginal_list):
        clique = clique_marginal_list[idx+1]
        # print(f"start = {start}, clique = {clique}")
        if len(finished_attr) == 0:
            cond_attr = common_attr
            for attr in start:
                print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))
                df = pandas_generate_cond_column_data(df, model,
                    clique_marginal[start], cond_attr, attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)

        separator = set(start) & set(clique)
        print('start: {}, clique: {}, sep: {}'.format(start, clique, separator))
        cond_attr = list(separator)
        for attr in clique:
            if attr not in finished_attr:
                print('  cond_attr: {}, attr: {} {}/{}'.format(cond_attr, attr, len(finished_attr), len(all_attr_list)))
                df = pandas_generate_cond_column_data(df, model,
                    clique_marginal[clique], cond_attr,
                    attr, all_attr_list)
                finished_attr.add(attr)
                cond_attr.append(attr)
        if idx == len(clique_marginal_list) - 2:
            break
    return df

def concat(num_party=2, data_name="acs", exp_name="test"):
    _, all_attr_list = read_csv('./data/' + data_name + '.csv')

    # model = MarkovRandomField.load_model('./temp/party_' + data_name + '_model.mrf')
    # data_list = model.synthetic_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
    # marginal_exp([data_list], data_name)

    parties = [chr(65+i) for i in range(num_party)] # A,B,...Z
    models = []
    for party in parties:
        print(f"# Distirbution from Party {party} #")
        models.append(MarkovRandomField.load_model(f'./temp/{exp_name}_party{party}_{data_name}_model.mrf'))

    data_list = synthetic(models)
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
    print("Evaluation of mutual information =", diff/num_pairs)

if __name__ == '__main__':
    data_name = "dummy_12_8_100000"
    data_list = concat(num_party=2, data_name=data_name)
    marginal_exp([data_list], data_name)
