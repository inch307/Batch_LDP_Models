# N-output Mechanism: Estimating Statistical Information from Numerical Data under Local Differential Privacy

This repository contains the implementation of the N-output mechanism for estimating various statistical information such as mean, distribution, variance, and quantile under Local Differential Privacy (LDP). The code is based on the paper "N-output Mechanism: Estimating Statistical Information from Numerical Data under Local Differential Privacy," which is currently under submission to VLDB 2025.

## Overview
- **mean_estimation.py**: Experiments for mean estimation.
- **dist_estimation.py**: Experiments for distribution, variance, and quantile estimation.
- **train.py**: Experiments for a case study on federated learning.

## Reproducing the Experiments

You can reproduce the experiments described in the paper using the following commands:

### Mean Estimation
```bash
python mean_estimation.py --num 100 --mech nopm_a3m --eps 4 --dataset ny_time_2023
python mean_estimation.py --num 100 --mech nopm_a3m --eps 4 --dataset hpc_voltage
python mean_estimation.py --num 100 --mech nopm_a3m --eps 4 --dataset beta
```

### Distribution, Variance, and Quantile Estimation
```bash
python dist_estimation.py --num 100 --mech nopm --eps 4 --dataset ny_time_2023
python dist_estimation.py --num 100 --mech nopm --eps 4 --dataset hpc_voltage
python dist_estimation.py --num 100 --mech nopm --eps 4 --dataset beta
```

### Case Study on Federated Learning
```bash
python train.py --dataset gamma --mech nonprive --eps 5 --weight_decay 0.001
python train.py --dataset gamma --mech nopm --eps 5
python train.py --dataset shuttle --mech nopm --eps 5 --weight_decay 0.001
python train.py --dataset shuttle --mech nopm --eps 5
```
