import numpy as np
import math

import topm
import n_output
import tpem
import data_load_numerical
import argparse

from utils_dis import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from a3m_dis import opt_variance, a3m_perturb

def eq(a, b, eps=1e-6):
    return abs(a - b) < eps

def create_beta_data(data_shape, a, b):
    data = np.random.beta(a=a, b=b, size=data_shape)
    data = 2*data - 1
    # mean: a / (a+b)
    mean = 2*(a / (a+b))-1
    var = 4 * a* b / (a+b)**2 / (a+b+1)
    return data, mean, var

def get_mech(name, d, eps, rng):
    k = max(1,min(d, math.floor(eps/2.5)))
    if name == 'duchi':
        return n_output.NOUTPUT(d, eps, rng, 2, k=1)
    elif name == 'pm_sub':
        return topm.TOPM(d, eps, rng, k=k)
    elif name == 'to' or name == 'topm':
        if eps > 1.7:
            return n_output.NOUTPUT(d, eps, rng, 3, k=k)
        else:
            return topm.TOPM(d, eps, rng, k=k)
    elif name == 'no' or name == 'nopm' or name == 'nopm_a3m':
        return n_output.NOUTPUT(d, eps, rng, None, k=1)

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

# 2PEM config
parser.add_argument('--tol1', type=float, help='tau1 for EM', default=1e-5)
parser.add_argument('--tol2', type=float, help='tau2 for EM', default=1e-5)
parser.add_argument('--sig', type=float, help='1/lambda^2', default=1)

## aaa
# range for DATA
parser.add_argument("--range", help="range for data", type=float, default=1)
parser.add_argument("--bin_size", help="bin length", type=float, default=0.125)
parser.add_argument("--axRatio", help="ratio between amax/xmax", type=float, default=4)
parser.add_argument("--s", help="split ratio", type=float, default=0.1)

args = parser.parse_args()
rng = np.random.default_rng(seed=args.seed)
np.random.seed(args.seed)

args = parser.parse_args()
print(args)
num_exp = args.num
name = args.mech

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

data01 = (data - (-1)) / (2)
true_mean01 = (true_mean + 1) / 2
true_var01 = (true_var) / 4

## a3m
split_ratio = args.s # proportion of data for frequency estimation for 3am


mean_squared_lst = []
rng = np.random.default_rng(seed=args.seed)
np.random.seed(args.seed)

mech = get_mech(args.mech, d, eps, rng)
mean_err_sum = 0
if args.mech == 'a3m':
    for n in range(num_exp):
        inds = np.argwhere(rng.random(len(data)) < (1/d))
        sampled_data = data[inds]
        a3m_n = len(sampled_data)

        # split for 3am
        data_1 = sampled_data[0:int(split_ratio*a3m_n)]
        data_2 = sampled_data[int(split_ratio*a3m_n):a3m_n]


        true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.range, args.range, args.bin_size, eps)    
        true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.range, args.range, args.bin_size, eps)    
        # convert to frequency
        true_freq = histogram_to_freq(true_histogram_2, -args.range, args.range, args.bin_size)
        noisy_freq = histogram_to_freq(noisy_histogram_1, -args.range, args.range, args.bin_size)
        # denoise the histogram and convert to frequency
        estimated_freq = denoise_histogram_RR(noisy_histogram_1, -args.range, args.range, args.bin_size, eps)
        """ pure """
        # use estimated freq to generate a3m noise
        noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.range, args.bin_size, args.axRatio, eps, 0)
        # perturb with a3m
        a3m_noise_pure = a3m_perturb(true_histogram_2, args.range, args.bin_size, noise_values, opt_distribution_pure)
        M = estimated_freq.size
        x_grid = -args.range + np.array(range(M)) * args.bin_size
        # print(x_grid)
        clean_dis_mean = np.sum(x_grid * true_freq)
        mean_err_sum += np.power(clean_dis_mean+np.sum(a3m_noise_pure) / (a3m_n-int(split_ratio*a3m_n))-true_mean, 2)

elif args.mech == 'nopm_a3m':
    for n in range(num_exp):
        inds = np.argwhere(rng.random(len(data)) < (1/d))
        sampled_data = data[inds]
        a3m_n = len(sampled_data)

        # split for 3am
        data_1 = sampled_data[0:int(split_ratio*a3m_n)]
        data_2 = sampled_data[int(split_ratio*a3m_n):a3m_n]

        true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.range, args.range, args.bin_size, eps)  
        true_freq = histogram_to_freq(true_histogram_2, -args.range, args.range, args.bin_size) 

        noisy_output, no_pm_inds = mech.HM_batch(data_1)
        nopm_mean = np.mean(noisy_output)
        probpm = tpem.ProbPM_LR(len(true_freq), 1, eps)
        probnm = tpem.ProbNM(len(true_freq), mech)
        no_output = noisy_output[no_pm_inds[0]]
        pm_output = noisy_output[no_pm_inds[1]]
        pm_perturbed_hist = np.histogram(pm_output, probpm.noisy_d, [-mech.pm.A, mech.pm.A])
        pm_perturbed_hist = pm_perturbed_hist[0] / np.sum(pm_perturbed_hist[0])
        no_perturbed_hist = np.zeros(mech.best_hybrid_mech['N'])
        for i in range(mech.best_hybrid_mech['N']):
            no_perturbed_hist[i] = np.count_nonzero(eq(no_output, mech.best_hybrid_mech['ais'][i]))
        no_perturbed_hist = no_perturbed_hist / np.sum(no_perturbed_hist)
        pm_est_hist = tpem.EM(probpm.prob, len(true_freq), pm_perturbed_hist, args.tol1)
        no_est_hist = tpem.EM_MAP(probnm.prob, len(true_freq), no_perturbed_hist, pm_est_hist, args.sig, args.tol2)
        estimated_freq = no_est_hist / np.sum(no_est_hist)

        true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.range, args.range, args.bin_size, eps)    
        noisy_freq = denoise_histogram_RR(noisy_histogram_1, -args.range, args.range, args.bin_size, eps)
        true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.range, args.range, args.bin_size, eps)    
        """ pure """
        # use estimated freq to generate a3m noise
        noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.range, args.bin_size, args.axRatio, eps, 0)
        # perturb with a3m
        a3m_noise_pure = a3m_perturb(true_histogram_2, args.range, args.bin_size, noise_values, opt_distribution_pure)
        M = estimated_freq.size
        x_grid = -args.range + np.array(range(M)) * args.bin_size
        # print(x_grid)
        clean_dis_mean = np.sum(x_grid * true_freq)
        mean = (clean_dis_mean*(a3m_n-int(split_ratio*a3m_n)) + np.sum(a3m_noise_pure) + np.sum(noisy_output)) / a3m_n
        mean_err_sum += (mean - true_mean)**2

else:
    for n in range(num_exp):
        noisy_output, _ = mech.exp_multi(args.mech, data)
        mean_err_sum += (np.mean(noisy_output) - true_mean)**2

MSE = (mean_err_sum / num_exp)

print(f'Dataset:{args.dataset}, Mechanism:{args.mech}, Mean estimation MSE: {MSE}, RMSE: {np.sqrt(MSE)}')
