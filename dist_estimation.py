import numpy as np
import os
import pandas as pd
import math

import data_load_numerical
import tpem
import topm
import n_output
import argparse
from scipy.stats import wasserstein_distance
import square_wave
import cfo

# Distribution accuracy one dim
def eq(a, b, eps=1e-6):
    return abs(a - b) < eps

def create_beta_data(data_shape, a, b):
    data = np.random.beta(a=a, b=b, size=data_shape)
    data = 2*data - 1
    # mean: a / (a+b)
    mean = 2*(a / (a+b))-1
    var = 4 * a* b / (a+b)**2 / (a+b+1)
    return data, mean, var

def get_mech(name, d, eps, rng, bins):
    if name == 'nopm' in name:
        return n_output.NOUTPUT(d, eps, rng, None, k=1)
    elif name == 'sw':
        return square_wave.SW(d, eps, rng)
    elif name == 'de':
        return cfo.DE(d, eps, bins, rng)
    elif name == 'oue':
        return cfo.OUE(d, eps, bins, rng)
    
    
def quantiles_from_hist(hist, bin_vlaues):
    quantiles = [0] * 9
    # print(quantiles)

    sum = 0
    cdf = []
    for i in range(len(hist)):
        sum += hist[i]
        cdf.append(sum)

    for i in range(9):
        q = 0.1 * (i+1)
        for k, v in enumerate(cdf):
            if v >= q:
                quantiles[i] = bin_vlaues[k]
                break

    return np.array(quantiles)
    
parser = argparse.ArgumentParser()
# Beta distribution cofig
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--sam', type=int, default=100000)
parser.add_argument('--a', type=float, default=3)
parser.add_argument('--b', type=float, default=2)

# Experiment config
parser.add_argument('--dataset', help='beta, ny_time_2023, hpc_voltage')
parser.add_argument('--seed', type = int, default=2025)
parser.add_argument('--num', type=int, help='the number of runs', default=100)
parser.add_argument('--mech', help='duchi -> duchi, no -> n_output, nopm -> HM-NP, to -> three-output, topm -> HM-TP, a3m -> AAA, nopm_a3m -> NP-AAA, pm_sub -> PM-sub')
parser.add_argument('--eps', type=float, help='the privacy budget')
parser.add_argument('--bins', type=int, default=1024)

# 2PEM config
parser.add_argument('--tol1', type=float, help='tau1 for EM', default=1e-5)
parser.add_argument('--tol2', type=float, help='tau2 for EM', default=1e-5)
parser.add_argument('--sig', type=float, help='1/lambda^2', default=1)


args = parser.parse_args()
print(args)
num_exp = args.num

rng = np.random.default_rng(seed=args.seed)
np.random.seed(args.seed)

if args.dataset == 'ny_time_2023':
    data, true_mean, true_var, data_shape = data_load_numerical.load_ny_datetime_2023()
    # print(f' mean: {true_mean}, var: {true_var}')
    num_sam, d = data_shape
 
elif args.dataset == 'beta':
    num_sam = args.sam
    d = args.dim
    data, true_mean, true_var = create_beta_data(num_sam, args.a, args.b)

elif args.dataset == 'hpc_voltage':
    data, true_mean, true_var, data_shape = data_load_numerical.load_hpc_voltage()
    num_sam, d = data_shape

eps = args.eps

true_hist, bins = np.histogram(data, args.bins, [-1, 1])
true_hist = true_hist / true_hist.sum()
discretized_data = np.digitize(data, bins[1:-1])
center_bins = (bins[0:-1] + bins[1:]) / 2
true_ex2 = true_var + true_mean**2
true_quantile = quantiles_from_hist(true_hist, center_bins)

data01 = (data - (-1)) / (2)
true_mean01 = (true_mean + 1) / 2
true_var01 = (true_var) / 4
true_ex201 = true_ex2 / 4
true_hist01, bins01 = np.histogram(data, args.bins, [0, 1])
center_bins01 = (bins01[0:-1] + bins01[1:]) / 2
true_quantile01 = quantiles_from_hist(true_hist01, center_bins01)


mech = get_mech(args.mech, d, eps, rng, args.bins)
wass_sum = 0
var_sum = 0
quantile_sum = 0
if args.mech == 'nopm':
    probpm = tpem.ProbPM_LR(args.bins, mech.k, eps)
    probnm = tpem.ProbNM(args.bins, mech)

    for n in range(num_exp):
        noisy_output, no_pm_inds = mech.HM_multi_simul_batch(data)
        nopm_mean = np.mean(noisy_output)
        noisy_output = noisy_output * mech.k / mech.d
        no_output = noisy_output[no_pm_inds[0]]
        pm_output = noisy_output[no_pm_inds[1]]
        pm_no_output = noisy_output[np.concatenate(no_pm_inds)]
        pm_perturbed_hist = np.histogram(pm_output, probpm.noisy_d, [-mech.pm.A, mech.pm.A])
        pm_perturbed_hist = pm_perturbed_hist[0] / np.sum(pm_perturbed_hist[0])
        no_perturbed_hist = np.zeros(mech.best_hybrid_mech['N'])
        for i in range(mech.best_hybrid_mech['N']):
            no_perturbed_hist[i] = np.count_nonzero(eq(no_output, mech.best_hybrid_mech['ais'][i]))
        no_perturbed_hist = no_perturbed_hist / np.sum(no_perturbed_hist)

        # pm first
        pm_est_hist_pf = tpem.EM(probpm.prob, args.bins, pm_perturbed_hist, args.tol1)
        no_est_hist_pf = tpem.EM_MAP(probnm.prob, args.bins, no_perturbed_hist, pm_est_hist_pf, args.sig, args.tol2)
        no_est_hist_pf = no_est_hist_pf / np.sum(no_est_hist_pf)
        wass_sum += wasserstein_distance(true_hist, no_est_hist_pf)
        var_sum += ((np.inner(no_est_hist_pf, center_bins**2) - nopm_mean**2) - true_var)**2
        est_quantile = quantiles_from_hist(no_est_hist_pf, center_bins)
        quantile_sum += np.mean((true_quantile - est_quantile)**2)

elif args.mech == 'sw':
    for n in range(num_exp):
        sw_output, inds = mech.SW_batch_multi_simul(data01)
        sw_output = sw_output[inds]

        sw_perturbed_hist = np.histogram(sw_output, args.bins, [-mech.b, 1+mech.b])
        sw_perturbed_hist = sw_perturbed_hist[0] / np.sum(sw_perturbed_hist[0])
        sw_est_hist = tpem.EM(mech.get_prob(args.bins, args.bins), args.bins, sw_perturbed_hist, args.tol1)
        sw_mean = np.inner(sw_est_hist, center_bins)
        var_sum += ((np.inner(sw_est_hist, center_bins**2) - sw_mean**2) - true_var)**2
        wass_sum += wasserstein_distance(true_hist, sw_est_hist)
        est_quantile = quantiles_from_hist(sw_est_hist, center_bins)
        quantile_sum += np.mean((true_quantile - est_quantile)**2)

elif args.mech == 'de' or args.mech == 'oue':
    for n in range(num_exp):
        est_hist = mech.batch_multi_simul(discretized_data)
        norm_hist = est_hist / np.sum(est_hist)
        wass_sum += wasserstein_distance(true_hist, norm_hist)
        cfo_mean = np.inner(norm_hist, center_bins)
        var_sum += ((np.inner(norm_hist, center_bins**2) - cfo_mean**2) - true_var)**2
        est_quantile = quantiles_from_hist(norm_hist, center_bins)
        quantile_sum += np.mean((true_quantile - est_quantile)**2)
        
wass = wass_sum / num_exp
var_err = var_sum / num_exp
quantile_err = quantile_sum / num_exp

print(f'Dataset:{args.dataset}, Mechanism:{args.mech}')
print(f'Wasserstein distance: {wass}, Variance: {var_err}, quantiles: {quantile_err}')
