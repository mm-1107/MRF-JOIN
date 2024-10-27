# MRF-JOIN: Differentially Private Vertical Data Synthesis via Federated Marginal Join on Shared Attributes


## 1. Setup

Requirements:
- Python 3.9
- CUDA 13.2.0
- GPU that supports `cupy`

Install dependencies:

```
pip3 install -r requirements.txt
```

## 2. Usage

### 2.1 Reproduce Paper Results

To reproduce the experimental results from the paper:

```
python3 script.py --dataset=[nltcs/acs/br2000/adult] --epsilon=[0.4/0.8/1.6/3.2] --task=[TVD/SVM]
```

### 2.2. Multiple parties
For multi-party data synthesis, the model can be run as Star Model or Chain Model as shown below:

```
python3 script.py --dataset=[nltcs/acs/br2000/adult] --party=[2/4/8] --chain=[True/False]
```

## 3. Privacy Budget

Calculate privacy budgets for different ε/δ using `cal_privacy_budget()` in `PrivMRF/utils/tools.py`.

## Code Reference

https://github.com/caicre/PrivMRF.git
