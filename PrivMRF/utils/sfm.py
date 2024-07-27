import numpy as np
import math
import random
import xxhash
import matplotlib.pyplot as plt
import os
import csv
import json
def read_csv(path, print_info=True):
    if print_info:
        print('    read csv data ' + path)
    data_list = []
    with open(path, 'r') as in_file:
        reader = csv.reader(in_file)
        headings = next(reader)
        data_list = [line for line in reader]
    return data_list, headings

def read_json_domain(domain_path):
    domain = json.load(open(domain_path, 'r'))
    domain = {int(attr): domain[attr] for attr in domain}
    return domain

class Sketch(object):
    """docstring for SFM."""

    def __init__(self, k, data):
        super(Sketch, self).__init__()
        self.k = k
        self.precision = 24
        self.bucket = 2 ** self.k
        self.sketch = self.pcsa(data)

    def pcsa(self, data):
        sketch = np.zeros((self.bucket, self.precision), dtype=int)
        hist = [0]*self.bucket
        for x in data:
            hashed_x = xxhash.xxh64(x).hexdigest()
            # hex = hashlib.sha256(x).hexdigest()
            hashed_x = format(int(hashed_x, 16), '064b')
            i = int(hashed_x[:self.k], 2) # value of first k bits x_0, x_1,..., x_k-1
            # i = hashed_x % self.bucket
            # hist[i] += 1
            b = int(hashed_x[self.k:], 2)   # x_k, x_k+1, ...
            # https://stackoverflow.com/questions/18806481/how-can-i-get-the-value-of-the-least-significant-bit-in-a-number
            j = min(self.precision-1, int(np.log2(b & -b)))    # the index of the least significant 1 in b

            #     print(sketch[i,j], hashed_x[:self.k], hashed_x[self.k:])
            sketch[i,j] = 1
        return sketch

class noisySketch(object):
    """docstring for noisySketch."""

    def __init__(self, eps, sketch):
        super(noisySketch, self).__init__()
        self.eps = eps
        self.bucket = sketch.shape[0]
        self.precision = sketch.shape[1]
        self.noisy_sketch = self.randomize(sketch)
    # bit-flipping algorithm
    def randomize(self, raw):
        noisy_sketch = np.zeros((self.bucket, self.precision), dtype=int)
        p = math.exp(self.eps) / (math.exp(self.eps)+1)
        for i in range(self.bucket):
            for j in range(self.precision):
                r1 = np.random.rand(1)
                if r1 < p:
                    noisy_sketch[i,j] = raw[i,j]
                else:
                    noisy_sketch[i,j] = 1 - raw[i,j]
                # noisy_sketch[i,j] = sketch[i,j] if ber == 1 else 1 - sketch[i,j]
        return noisy_sketch

def likelihood(n, sketch, p, q):
    bucket = sketch.shape[0]
    precision = sketch.shape[1]
    sum = 0
    for j in range(precision):
        # pho =2^-min(j, P-1)/B, gamma_n = (1-pho)^n
        gamma = 1 - pow(2, -min(j+1, precision))/bucket
        gamma_n = pow(gamma, n)
        tmp = (p-q)*gamma_n
        # print(f"n={n}, gamma = {gamma}, gamma_n = {gamma_n}, (p-q)*gamma_n = {tmp}")
        tmp_1 = math.log(1 - p + tmp)
        tmp_2 = math.log(p - tmp)
        for i in range(bucket):
            t_ij = sketch[i,j]
            sum += (1 - t_ij)*tmp_1 + t_ij*tmp_2
    return sum

def first_derivative(n, sketch, p, q):
    bucket = sketch.shape[0]
    precision = sketch.shape[1]
    sum = 0
    for j in range(precision):
        # pho =2^-min(j, P-1)/B, gamma_n = (1-pho)^n
        gamma = 1 - pow(2, -min(j+1, precision))/bucket
        gamma_n = pow(gamma, n)
        log_gamma =  math.log(gamma)
        tmp_top = (p - q) * gamma_n * log_gamma
        tmp_bottom = (p - q) * gamma_n
        tmp_1 = tmp_top / (1 - p + tmp_bottom)
        tmp_2 = tmp_top / (p - tmp_bottom)
        for i in range(bucket):
            t_ij = sketch[i,j]
            sum += (1 - t_ij)*tmp_1 - t_ij*tmp_2
    return sum

def derivative(n, sketch, p, q):
    bucket = sketch.shape[0]
    precision = sketch.shape[1]
    first = second = 0
    for j in range(precision):
        # pho =2^-min(j, P-1)/B, gamma_n = (1-pho)^n
        gamma = 1 - 1 / (pow(2, min(j+1, precision)) * bucket)
        gamma_n = pow(gamma, n)
        log_gamma =  math.log(gamma)
        tmp_top = (p - q) * gamma_n * log_gamma
        tmp_bottom = (p - q) * gamma_n
        first_1_bottom = 1 - p + tmp_bottom
        first_2_bottom = p - tmp_bottom
        first_1 = tmp_top / first_1_bottom
        first_2 = tmp_top / first_2_bottom
        second_1 = ((1 - p) * log_gamma * tmp_top) / (first_1_bottom * first_1_bottom)
        second_2 = (p * log_gamma * tmp_top) / (first_2_bottom * first_2_bottom)
        for i in range(bucket):
            t_ij = sketch[i,j]
            first += (1 - t_ij) * first_1 - t_ij * first_2
            second += (1 - t_ij) * second_1 - t_ij * second_2
    return first, second

def determistic_estimate(sketch):
    z_ = 0
    for i in range(skch.sketch.shape[0]):
        for j in range(skch.sketch.shape[1]):
            if skch.sketch[i,j] == 0:
                z_ += j
                # print(f"{z_}, sketch[{i},{j}]")
                break
    z_ = z_ / skch.sketch.shape[0]
    z_ = skch.sketch.shape[0] * pow(2, z_)/ 0.77351
    print(f"z = {z_}")
    return z_

def est_error(skch, n):
    p = math.exp(skch.eps) / (math.exp(skch.eps)+1)
    q = 1 - p
    sum = 0
    for j in range(skch.precision):
        gamma = 1 - 1 / (pow(2, min(j+1, skch.precision)) * skch.bucket)
        gamma_n = pow(gamma, n)
        log_gamma =  math.log(gamma)
        f1 = p / (p - (p - q) * gamma_n)
        f2 = (1 - p) / (1 - p + (p - q) * gamma_n)
        sum += log_gamma ** 2 * gamma_n * (f1 - f2)
    est_error = 1 / np.sqrt(skch.bucket*(p-q)*sum) / n
    # print(f"B={self.bucket}, P={self.precision}, n={n}, Estimated error: {est_error}")
    return est_error

def estimate(eps, sketch):
    # Newton–Raphson method
    p = math.exp(eps) / (math.exp(eps)+1)
    q = 1 - p
    n = 1000
    stop = 1
    # print(n, likelihood(n, sketch, p, q))
    cnt = 0
    while(stop > 0.001 and cnt < 100):
        first, second = derivative(n, sketch, p, q)
        stop = first / second
        n = n - stop
        stop = abs(stop)
        cnt += 1
        # print(n, self.likelihood(n, sketch, p, q))
    # print(n)
    return n

def prob_eps(eps):
    return 1 / (math.exp(eps) + 1)

def merge(s1, s2, eps_1, eps_2):
    s_merge = np.zeros_like(s1, dtype=int)
    q_eps_1 = prob_eps(eps_1)
    q_eps_2 = prob_eps(eps_2)
    eps_merge = -math.log(math.exp(-eps_1)+math.exp(-eps_2)-math.exp(-eps_1-eps_2))
    q_merge =  prob_eps(eps_merge)
    K1 = np.array([[1-q_eps_1, q_eps_1], [q_eps_1, 1-q_eps_1]])
    K2 = np.array([[1-q_eps_2, q_eps_2], [q_eps_2, 1-q_eps_2]])
    v = np.array([q_merge, q_merge, q_merge, 1 - q_merge])
    # K_merge = [t00, t01, t10, t11]
    K_merge = np.kron(np.linalg.inv(K1), np.linalg.inv(K2)) @ v
    # print("e* =", eps_merge, K_merge)
    t = np.zeros((2,2))
    t[0,0] = K_merge[0]
    t[0,1] = K_merge[1]
    t[1,0] = K_merge[2]
    t[1,1] = K_merge[3]
    # print(K_merge)
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            p = t[s1[i, j]][s2[i, j]]
            r1 = np.random.rand(1)
            if r1 < p:
                s_merge[i,j] = 1
    return eps_merge, s_merge

def RRMSE(n, n_est, trial):
    error = 0
    for n_ in n_est:
        error += (n - n_) ** 2
    error = np.sqrt(error / trial) / n
    print("RRMSE:", error)

def test(k=12, n=10**4, eps=1):
    data = np.arange(n)
    trial = 1
    n_true = len(set(data))
    print("The number of unique", n_true)
    print("Epsilon", eps)
    n_est = []
    for t in range(trial):
        skch = Sketch(k=k, data=data)
        print(f"P = {skch.sketch.shape[1]}, B = {skch.sketch.shape[0]}")
        nskch = noisySketch(eps, skch.sketch)
        n_est.append(estimate(eps, nskch.noisy_sketch))
    RRMSE(n_true, n_est, trial)

def test_merge(k=12, n=10**6, eps=1):
    print("=== Test merging sketches ===")
    data = np.arange(n)
    data_1 = np.arange(n)
    n_true = len(set(data) & set(data_1))
    trial = 1
    print("The number of unique", n_true)
    print("Epsilon", eps)
    n_est = []
    for t in range(trial):
        sketch_1 = Sketch(k=k, data=data)
        sketch_2 = Sketch(k=k, data=data_1)
        # print(f"P = {sketch_1.shape[1]}, B = {sketch_1.shape[0]}")
        nskch_1 = noisySketch(eps, sketch_1.sketch)
        nskch_2 = noisySketch(eps, sketch_2.sketch)
        eps_merge, s_merge = merge(nskch_1.noisy_sketch, nskch_2.noisy_sketch, eps, eps)
        print(f"eps1={eps}, eps2={eps}, eps_merge={eps_merge}")
        eps_merge_2, s_merge = merge(s_merge, s_merge, eps_merge, eps_merge)
        print(f"eps1={eps_merge}, eps2={eps_merge}, eps_merge={eps_merge_2}")
        n_est.append(estimate(eps_merge, s_merge))
    RRMSE(n_true, n_est, trial)

def plot_error(eps):
    for n in [10**2, 10**3, 10**4, 10**5, 10**6]:
        height = []
        for B in [10, 11, 12, 13, 14]:
            sfm = SFM(k=B, eps=eps)
            error = sfm.est_error(n)
            height.append(error)
        plt.plot(["10", "11", "12", "13", "14"], height, marker="o", label=n)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"epsilon={eps}")
    plt.savefig(f"error_{eps}.png")

def LocEnc(eps, data, attr_list, json_domain):
    locenc_dict = dict()
    k = 10
    print(f"---> {len(data)}, eps={eps}, k={k}")
    # skech_party = np.zeros((self.bucket, self.precision), dtype=int)
    for attr in attr_list:
        domain = json_domain[attr]["domain"]
        col = data[:,attr]
        # print(attr, col, len(col), domain)
        sketch_by_domain = dict()
        for d in range(domain):
            ids = np.where(col == d)[0]
            skch = Sketch(k=k, data=ids)
            nskch = noisySketch(eps, skch.sketch)
            sketch_by_domain[d] = nskch.noisy_sketch
            # estimate(eps, nskch.noisy_sketch)
            # print(f"--> {attr, d} correct =", np.count_nonzero(data[:,attr] == d))
        locenc_dict[attr] = sketch_by_domain
    return locenc_dict


def CarEst(data_path, json_domain_path, eps):
    data, _ = read_csv(data_path)
    data = np.array(data, dtype=int)
    json_domain = read_json_domain(json_domain_path)
    # If LocEnc ensures ε-DP, CarEst ensures εd-DP
    eps = eps / len(data[0])
    # Divide attrs
    attrs = list(json_domain.keys())
    attr_num = len(attrs)
    random.shuffle(attrs)
    attr_partyA = attrs[:int(attr_num/2)]
    attr_partyB = attrs[int(attr_num/2):]
    print(attr_partyA, attr_partyB)
    carest = dict()
    # Sketch for each attribute
    locenc_A = LocEnc(eps, data, attr_partyA, json_domain)
    locenc_B = LocEnc(eps, data, attr_partyB, json_domain)
    all_sketch = dict()
    # 1-d cardinality
    for attr in attr_partyA:
        all_sketch[attr] = locenc_A[attr]
        hist = np.zeros(len(locenc_A[attr]))
        for domain, skch in locenc_A[attr].items():
            hist[domain] = estimate(eps, skch)
        carest[tuple([attr])] = hist
    for attr in attr_partyB:
        all_sketch[attr] = locenc_B[attr]
        hist = np.zeros(len(locenc_B[attr]))
        for domain, skch in locenc_B[attr].items():
            hist[domain] = estimate(eps, skch)
            print(f"--> 1 way estimate = {hist[domain]}")
            ids = np.where(data[:,attr] == domain)
            print("--> 1 way correct =", len(ids[0]))
        carest[tuple([attr])] = hist
    all_sketch["eps"] = eps

    # Merge attr_partyA & attr_partyA
    for idx, attrA in enumerate(attr_partyA):
        for attrB in attr_partyA[idx+1:]:
            pair = tuple(sorted([attrA, attrB]))
            if pair[0] == attrA:
                first_sketch = locenc_A[attrA]
                second_sketch = locenc_A[attrB]
            else:
                first_sketch = locenc_A[attrB]
                second_sketch = locenc_A[attrA]
            hist = np.zeros((len(first_sketch), len(second_sketch)))
            for domain_1, skch_1 in first_sketch.items():
                for domain_2, skch_2 in second_sketch.items():
                    eps_merge, s_merge = merge(skch_1, skch_2, eps, eps)
                    hist[domain_1, domain_2] = estimate(eps_merge, s_merge)
                    # print(f"-> {pair}-{domain_1, domain_2}")
                    # print(f"--> estimate = {hist[domain_1, domain_2]}")
                    # ids = np.where(data[:,attrA] == domain_1)
                    # ids_ = np.where(data[:,attrB] == domain_2)
                    # print("--> correct =", len(np.intersect1d(ids[0],ids_[0])))
                    # print(f"----> {len(ids[0])}, {len(ids_[0])}")
            carest[pair] = hist
    # Merge attr_partyB & attr_partyB
    for idx, attrA in enumerate(attr_partyB):
        for attrB in attr_partyB[idx+1:]:
            pair = tuple(sorted([attrA, attrB]))
            if pair[0] == attrA:
                first_sketch = locenc_B[attrA]
                second_sketch = locenc_B[attrB]
            else:
                first_sketch = locenc_B[attrB]
                second_sketch = locenc_B[attrA]
            hist = np.zeros((len(first_sketch), len(second_sketch)))
            for domain_1, skch_1 in first_sketch.items():
                for domain_2, skch_2 in second_sketch.items():
                    eps_merge, s_merge = merge(skch_1, skch_2, eps, eps)
                    hist[domain_1, domain_2] = estimate(eps_merge, s_merge)
            carest[pair] = hist
    # Merge attr_partyA & attr_partyB
    for attrA in attr_partyA:
        for attrB in attr_partyB:
            pair = tuple(sorted([attrA, attrB]))
            if pair[0] == attrA:
                first_sketch = locenc_A[attrA]
                second_sketch = locenc_B[attrB]
            else:
                first_sketch = locenc_B[attrB]
                second_sketch = locenc_A[attrA]
            hist = np.zeros((len(first_sketch), len(second_sketch)))
            for domain_1, skch_1 in first_sketch.items():
                for domain_2, skch_2 in second_sketch.items():
                    eps_merge, s_merge = merge(skch_1, skch_2, eps, eps)
                    hist[domain_1, domain_2] = estimate(eps_merge, s_merge)
            carest[pair] = hist
    return carest, all_sketch

def histogramdd(index_list, bins):
    if len(index_list) < 3:
        # for m in range(shape[0]):
        #     carest[m] = estimate(eps, sketch[index_list[0]][m])
        # return carest
        carest = np.load('../preprocess/carest.npy', allow_pickle=True)
        carest = carest.item()
        return np.nan_to_num(carest[index_list])
    # elif len(index_list) == 2:
    #     for m in range(shape[0]):
    #         for n in range(shape[1]):
    #             eps_merge, s_merge = merge(sketch[index_list[0]][m], sketch[index_list[1]][n], eps, eps)
    #             carest[m, n] = estimate(eps_merge, s_merge)
    #     return carest
    elif len(index_list) == 3:
        sketch = np.load('../preprocess/all_sketch.npy', allow_pickle=True).item()
        eps = sketch["eps"]
        shape = tuple([len(value) - 1 for value in bins])
        carest = np.zeros(shape)
        for m in range(shape[0]):
            for n in range(shape[1]):
                eps_merge, s_merge = merge(sketch[index_list[0]][m], sketch[index_list[1]][n], eps, eps)
                for l in range(shape[2]):
                    eps_merge, s_merge = merge(s_merge, sketch[index_list[2]][l], eps_merge, eps)
                    carest[m, n, l] = estimate(eps_merge, s_merge)
                    # print("carest =", carest[m, n, l])
        return np.nan_to_num(carest)
    elif len(index_list) == 4:
        sketch = np.load('../preprocess/all_sketch.npy', allow_pickle=True).item()
        eps = sketch["eps"]
        shape = tuple([len(value) - 1 for value in bins])
        carest = np.zeros(shape)
        for m in range(shape[0]):
            for n in range(shape[1]):
                eps_merge, s_merge = merge(sketch[index_list[0]][m], sketch[index_list[1]][n], eps, eps)
                for l in range(shape[2]):
                    eps_merge, s_merge = merge(s_merge, sketch[index_list[2]][l], eps_merge, eps)
                    for k in range(shape[3]):
                        eps_merge, s_merge = merge(s_merge, sketch[index_list[3]][k], eps_merge, eps)
                        carest[m, n, l, k] = estimate(eps_merge, s_merge)
                    # print("carest =", carest[m, n, l])
        return np.nan_to_num(carest)

if __name__ == '__main__':
    # carest, all_sketch = sfm.CarEst("../../preprocess/nltcs.csv", "../../preprocess/nltcs.json", eps=960)
    # np.save('../../preprocess/carest.npy',  carest)
    # np.save('../../preprocess/all_sketch.npy',  all_sketch)

    data, _ = read_csv('../../preprocess/nltcs.csv')
    data = np.array(data, dtype=int)
    json_domain = read_json_domain('../../preprocess/nltcs.json')
    index_list = [1,2,3,4]
    shape = [json_domain[i]['domain'] for i in index_list]
    bins = [list(range(i+1)) for i in shape]
    print(histogramdd(tuple(index_list), bins))

    # print(bins)
    histogram, _ = np.histogramdd(data[:, index_list], bins=bins)
    print(f"np.histogramdd = {histogram}")
    ids = np.where(data[:,1] == 1)[0]
    ids_1 = np.where(data[:,2] == 0)[0]
    ids_2 = np.where(data[:,3] == 0)[0]
    ids_3 = np.where(data[:,4] == 1)[0]
    print("--> correct =", len(np.intersect1d(np.intersect1d(np.intersect1d(ids,ids_1),ids_2),ids_3)))
    # eps = 48.0
    # test_merge(k=10, n=10**4, eps=0.1)

    # carest = CarEst(data, json_domain, eps)
    # np.save('../../preprocess/carest.npy',  carest)

    # plot_error(eps=0.1)
    test(k=10, n=10**6, eps=1)
    test(k=10, n=10**6, eps=0.1)
    test(k=10, n=10**6, eps=0.05)
    test(k=10, n=10**6, eps=0.01)
    # test(k=12, n=10**6)
    # test(k=10, n=10**6)
