# Batch Local Differentially Private Models

This repository contains the implementation of Batch Local Differentially Private Models.

## Implemented Models
- **Duchi Model**
- **Three-ouput Model**
- **Piecewise Model**
- **Piecewise-sub optimal Model**
- **Hybrid(Duchi + Piecewise) Model**
- **Hybrid(Three-output + Piecewise-sub) Model**

## Overview
- **mean_estimation.py**: Experiments for mean estimation.
- **dist_estimation.py**: Experiments for distribution, variance, and quantile estimation.
- **train.py**: Experiments for a case study on federated learning.

## Dependencies
Ensure you have the necessary dependencies installed before running the experiments:
```bash
pip install -r requirements.txt
```
This code is compatible with Python 3.11.7

## Preparation: Dataset

Some datasets used in this project are large, so they are provided via external links. Please download the datasets from the following links:

- [NYC Taxi and Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [UCI Machine Learning Repository - Individual household electric power consumption Data Set](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

After downloading, place the datasets in the data directory.


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
