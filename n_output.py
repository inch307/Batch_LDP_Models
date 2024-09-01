import numpy as np
import math
import topm

from scipy.optimize import minimize
from scipy.optimize import root

class NOUTPUT():
    def __init__(self, d, eps, rng, N = None, range=1.0, k=1):
        self.d = d
        self.eps = eps
        self.rng = rng
        self.range = range
        self.k = k
        self.eps_k = self.eps / self.k

        # precomputed e
        self.ee = np.exp(self.eps_k)
        self.ee3 = np.exp(self.eps_k / 3)

        self.e = np.exp(self.eps)
        self.e3 = np.exp(self.eps / 3)

        # min maxVar
        if N is None:
            self.mechs = self.generate_mechanisms()
            self.best_mech = self.min_maxVar_mech()
        else:
            self.best_mech = self.construct_N(N)
            self.mechs = [self.best_mech]
            # self.best_hybrid_mech = self.construct_N
        self.A = (self.ee3 + 1) / (self.ee - 1)
        self.B = (self.ee3 + self.ee) * ((self.ee3 +1)**3 +self.ee -1) / (3 * self.ee3**2 * (self.ee - 1)**2)

        self.best_hybrid_mech = self.min_maxVar_hybrid_mech()

        self.pm = topm.TOPM(d, eps, rng, range, k)

    def construct_N(self, N):
        ais, p0 = construct_type0(self.eps_k, N)

        if len(ais) == 0:
            return None

        if N <= 3:
            max_var = worst_var(self.eps_k, ais, p0, N)
            return {'N': N, 'ais': ais, 'p0':p0, 'V': max_var, 'type': 0}

        V0 = Var_1(self.eps_k, ais, p0, N)
        V = Var_n(self.eps_k, ais, p0, N)
        if V >= V0:
            max_var = worst_var(self.eps_k, ais, p0, N)
        else:
            ais, p0 = construct_type1(self.eps_k, N)
            max_var = worst_var(self.eps_k, ais, p0, N)
            return {'N': N, 'ais': ais, 'p0':p0, 'V': max_var, 'type': 1}

    def min_maxVar_hybrid_mech(self):
        minV = np.inf
        best_mech = None

        A = (self.ee3 + 1) / (self.ee - 1)
        B = (self.ee3 + self.ee) * ((self.ee3 +1)**3 +self.ee -1) / (3 * self.ee3**2 * (self.ee - 1)**2)

        for mech in self.mechs:
            mech_copy = mech.copy()
            ais = mech['ais']
            N = mech['N']
            if N == 2:
                if self.eps_k < 0.610986:
                    alpha = 1
                else:
                    alpha = (self.ee3 + 1) / (self.ee + self.ee3)
                V = alpha * ((self.ee + 1) / (self.ee - 1))**2 + (1-alpha) * B
                if V < minV:
                    minV = V
                    mech_copy['alpha'] = alpha
                    mech_copy['V'] = V
                    best_mech = mech_copy
            else:
                x_star = (ais[-1] + ais[-2]) / 2
                p0 = mech['p0']
                if N % 2 == 0:
                    n = N / 2
                else:
                    n = (N-1) / 2
                P = (1-p0) / (self.ee + 2*n - 1)
                g1 = (4 * x_star**2 + 4*(1+A) * (np.sum(ais**2) * P - ais[-2] - B)) / (4*(1+A)**2)
                g2 = (A**2 * x_star**2) / ((1+A)**2)

                if g1 <= 0:
                    self.alpha = 1
                    x_H_star = worst_x(self.eps_k, ais, p0, N)
                else:
                    alpha1 = ((self.ee -1) * np.sqrt(g2 / g1) + self.ee3 + 1) / (self.ee + self.ee3)
                    alpha2 = (self.ee3 + 1) / (self.ee + self.ee3)
                    alpha3 = (self.ee3 + 1 ) / (self.ee3 + self.ee - x_star*(self.ee -1))
                    d = -A -B + np.sum(ais**2)*P + ais[-1] - 1

                    # print(f'alpha1: {alpha1}, alpha2: {alpha2}, alpha3: {alpha3}, d: {d}')

                    if alpha1 >= alpha3:
                        if alpha1 > 1:
                            if alpha3 > 1:
                                alpha = 1
                            else:
                                alpha = alpha3
                        else:
                            alpha = alpha1
                        x_H_star = alpha * (ais[-1] + ais[-2]) / (2 * (alpha + A*alpha - A))
                        # print('alpha1')
                    elif alpha2 < alpha1 and alpha1 < alpha3 and d <= 0:
                        if alpha3 > 1:
                            alpha = 1
                        else:
                            alpha = alpha3
                        x_H_star = alpha * (ais[-1] + ais[-2]) / (2 * (alpha + A*alpha - A))
                        
                        # print('alpha3')
                    elif alpha2 < alpha1 and alpha1 < alpha3 and d > 0:
                        # alpha = alpha2
                        alpha = 0
                        x_H_star = 1
                        # print('alpha2')

                V_star = alpha * (np.sum(ais**2)*P + (ais[-1] + ais[-2]) * x_H_star - ais[-2] - x_H_star**2) + (1-alpha)*(A*(x_H_star)**2 + B)
                V_1 = alpha * (np.sum(ais**2)*P + ais[-1] - 1) + (1-alpha)*(A + B)
                if V_1 > V_star:
                    x_H_star = 1
                    V = V_1
                else:
                    V = V_star

                # print(f'N: {mech_copy["N"]}, alpha: {alpha}, V: {V}')

                if V < minV:
                    minV = V
                    mech_copy['alpha'] = alpha
                    mech_copy['V'] = V
                    mech_copy['x_H_star'] = x_H_star
                    best_mech = mech_copy

        return best_mech
    
    def generate_mechanisms(self):
        mechs = []
        N_list, max_N = find_N(self.eps_k)

        # construct type 0
        for i, N in enumerate(N_list):
            ais, p0 = construct_type0(self.eps_k, N)

            if len(ais) > 0:
                max_var = worst_var(self.eps_k, ais, p0, N)
                if N % 2 == 0:
                    mechs.append({'N': N, 'ais': ais, 'p0':p0, 'V': max_var, 'type': 0})
                else:
                    if p0 != 0:
                        mechs.append({'N': N, 'ais': ais, 'p0':p0, 'V': max_var, 'type': 0})

        # construct type 1
        for i in range(max_N-3):
            N = max_N - i
            ais = []
            ais, p0 = construct_type1(self.eps_k, N)

            if len(ais) > 0:
                max_var = worst_var(self.eps_k, ais, p0, N)
                mechs.append({'N': N, 'ais': ais, 'p0':p0, 'V': max_var, 'type': 1})


        return mechs
    
    def min_maxVar_mech(self):
        best_mech = {'V': np.inf}
        for m in self.mechs:
            if m['V'] < best_mech['V']:
                best_mech = m
        
        return best_mech

    def get_max_var(self):
        return self.best_mech['V']


    def NO_batch(self, data):
        data = data.reshape(-1) / self.range
        ais = self.best_mech['ais']
        p0 = self.best_mech['p0']

        prob = prob_x(self.eps_k, ais, p0, data)
        cumulative_prob = np.cumsum(prob, axis=1)
        random_values = self.rng.random((prob.shape[0], 1))
        choices = (random_values < cumulative_prob).argmax(axis=1)

        return ais[choices] * self.range, None
    
    def NO_HM_batch(self, data):
        data = data.reshape(-1) / self.range
        ais = self.best_hybrid_mech['ais']
        p0 = self.best_hybrid_mech['p0']

        prob = prob_x(self.eps_k, ais, p0, data)
        cumulative_prob = np.cumsum(prob, axis=1)
        random_values = self.rng.random((prob.shape[0], 1))
        choices = (random_values < cumulative_prob).argmax(axis=1)
        
        return ais[choices] * self.range, None
    
    def NO_multi_batch(self, data):
        sampled_dims = np.random.choice(len(data), self.k, replace=False)
        noisy_output = np.zeros_like(data)

        noisy_output[sampled_dims], _ = self.NO_batch(data[sampled_dims])
        # no_inds = inds[no_pm_inds[0]]
        # pm_inds = inds[no_pm_inds[1]]
        return noisy_output * self.d / self.k
    
    def NO_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds], _ = self.NO_batch(data[inds])

        return noisy_output * self.d / self.k, None
    
    def HM_batch(self, data):
        data = data.reshape(-1)
        hm_random = self.rng.random(len(data))
        no_inds = np.argwhere(hm_random < self.best_hybrid_mech['alpha']).reshape(-1)
        pm_inds = np.argwhere(hm_random >= self.best_hybrid_mech['alpha']).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[no_inds], _ = self.NO_HM_batch(data[no_inds])
        noisy_output[pm_inds], _ = self.pm.PM_batch(data[pm_inds])

        return noisy_output, [no_inds, pm_inds]
    
    def HM_multi_batch(self, data):
        sampled_dims = np.random.choice(len(data), self.k, replace=False)
        noisy_output = np.zeros_like(data)

        noisy_output[sampled_dims], _ = self.HM_batch(data[sampled_dims])
        # no_inds = inds[no_pm_inds[0]]
        # pm_inds = inds[no_pm_inds[1]]
        return noisy_output * self.d / self.k
    
    def HM_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        noisy_output = np.zeros_like(data)

        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output[inds], no_pm_inds = self.HM_batch(data[inds])
        no_inds = inds[no_pm_inds[0]]
        pm_inds = inds[no_pm_inds[1]]
        return noisy_output * self.d / self.k , [no_inds, pm_inds]


    def exp_batch(self, name, data):
        if name == 'no':
            return self.NO_batch(data)
        elif name == 'to':
            return self.NO_batch(data)
        elif name == 'duchi':
            return self.NO_batch(data)
        elif name =='nopm':
            return self.HM_batch(data)
        elif name =='topm':
            return self.HM_batch(data)
    
    def exp_multi(self, name, data):
        if name == 'no':
            return self.NO_multi_simul_batch(data)
        elif name == 'to':
            return self.NO_multi_simul_batch(data)
        elif name == 'duchi':
            return self.NO_multi_simul_batch(data)
        elif name == 'nopm':
            return self.HM_multi_simul_batch(data)
        elif name == 'topm':
            return self.HM_multi_simul_batch(data)
        
    def multi(self, name, data):
        if name == 'no':
            return self.NO_multi_batch(data)
        elif name == 'to':
            return self.NO_multi_batch(data)
        elif name == 'duchi':
            return self.NO_multi_batch(data)
        elif name == 'nopm':
            return self.HM_multi_batch(data)
        elif name == 'topm':
            return self.HM_multi_batch(data)
            
def make_a(ais, N):
    rev_ais = list(reversed(ais))
    minus_ais = []
    for i in ais:
        minus_ais.append(-i)
    
    if N % 2 == 0:
        return np.array(minus_ais + rev_ais)
    else:
        return np.array(minus_ais + [0] + rev_ais)
    
def find_N(eps):
    N_list = [2, 3]
    max_N = 3

    for N in range(4, 1000):
        if N % 2 == 0:
            n = N / 2
            start_i = int(N/2)
        else:
            n = (N-1) / 2
            start_i = int((N+1)/2)

        ais, p0 = construct_type0(eps, N)

        if len(ais) > 0:
            max_N = N
        else:
            break

        V0 = Var_1(eps, ais, p0, N)
        V = Var_n(eps, ais, p0, N)
        if (V > V0) or math.isclose(V, V0, abs_tol=1e-9):
            N_list.append(N)

    return N_list, max_N

def check_sequence(ais):
    prev = ais[0]
    for ai in ais[1:]:
        if prev < ai:
            return False
    if ais[-1] < 0:
        return False
    return True

def ais_type0(eps, p0, N):
    ais = []
    if N % 2 == 0:
        n = N // 2
    else:
        n = (N-1) // 2
    P = (1-p0) / (np.exp(eps) + 2*n - 1)
    a_n = 1 / (np.exp(eps)- 1) / P
    
    ais.append(a_n)

    if N > 3:
        t = (math.exp(eps)-1)*P
        T = 4*t-2

        Pi = [0, 1]
        Qi = [1, 0]

        for i in range(n-2):
            Pi.append(T*Pi[-1] - Pi[-2])
            Qi.append(T*Qi[-1] - Qi[-2])
        del Pi[0]
        del Qi[0]
        Pi = np.array(Pi)
        Qi = np.array(Qi)

        a_n_1 = a_n * (2*t - 1 - 8 * P * np.inner(Pi, Qi)) / (1 + 8 * P * np.sum(Pi**2))
        ais.append(a_n_1)

        for i in range(n-2):
            ais.append(ais[-1]*T - ais[-2])
    valid_sequence = check_sequence(ais)
    if valid_sequence:
        ais = make_a(ais, N)
        return ais
    else:
        return []

def Var_1(eps, ais, p0, N):
    if N % 2 == 0:
        n = N / 2
        start_i = int(N/2)
    else:
        n = (N-1) / 2
        start_i = int((N+1)/2)

    P = (1-p0) / (np.exp(eps) + 2*n - 1)
    p_star = (1 - 2*(n-1)*P - np.exp(eps)*p0 ) / 2
    
    b = ais[start_i] * (np.exp(eps)*P +P - 2*p_star ) / ((np.exp(eps) -1)*P)
    max_point = min(1, b/2)
    prob = prob_x(eps, ais, p0, np.array([max_point]))
    return np.sum(prob * ais**2) - max_point**2

def Var_n(eps, ais, p0, N):
    if N % 2 == 0:
        n = N / 2
    else:
        n = (N-1) / 2
    P = (1-p0) / (np.exp(eps) + 2*n - 1)
    
    b = ais[-1] + ais[-2]
    max_point = min(1, b/2)
    prob = prob_x(eps, ais, p0, np.array([max_point]))
    return np.sum(prob * ais**2) - max_point**2

def worst_x(eps, ais, p0, N):
    if N % 2 == 0:
        n = N / 2
        start_i = int(N/2)
    else:
        n = (N-1) / 2
        start_i = int((N+1)/2)
    if N <= 3:
        P = (1-p0) / (np.exp(eps) + 2*n - 1)
        p_star = (1 - 2*(n-1)*P - np.exp(eps)*p0 ) / 2
        b = ais[start_i] * (np.exp(eps)*P +P - 2*p_star ) / ((np.exp(eps) -1)*P)
        max_point = min(1, b/2)
        return max_point
    else:
        b = ais[-1] + ais[-2]
        max_point = min(1, b/2)
        return max_point

def worst_var(eps, ais, p0, N):
    if N % 2 == 0:
        n = N / 2
        start_i = int(N/2)
    else:
        n = (N-1) / 2
        start_i = int((N+1)/2)
    if N <= 3:
        P = (1-p0) / (np.exp(eps) + 2*n - 1)
        p_star = (1 - 2*(n-1)*P - np.exp(eps)*p0 ) / 2
        
        b = ais[start_i] * (np.exp(eps)*P +P - 2*p_star ) / ((np.exp(eps) -1)*P)
        max_point = min(1, b/2)
        prob = prob_x(eps, ais, p0, np.array([max_point]))
        return np.sum(prob * ais**2) - max_point**2
    else:
        P = (1-p0) / (np.exp(eps) + 2*n - 1)
    
        b = ais[-1] + ais[-2]
        max_point = min(1, b/2)
        prob = prob_x(eps, ais, p0, np.array([max_point]))
        return np.sum(prob * ais**2) - max_point**2


def prob_x(eps, ais, p0, data):
    N = len(ais)
    ee = np.exp(eps)
    if N % 2 == 0:
        n = N / 2
        start_i = int(N/2)
        P = (1-p0) / (ee + 2*n - 1)
        p_star = (1 - 2*(n-1)*P - ee*p0 ) / 2

        prob = np.full((len(data), N), P)
        x_lst = np.array(ais) * (ee - 1) * P 
        bins = x_lst[1:-1] * (ee - 1) * P
        digitized = np.digitize(data, bins)

        for i in range(N-1):
            inds = np.argwhere(digitized == i).reshape(-1)

            data_inds = data[inds]
            prob[inds, i+1] = (data_inds + (ais[i+1] - ee*ais[i]) * P) / (ais[i+1] - ais[i])
            prob[inds, i] = ((ee*ais[i+1] - ais[i]) * P - data_inds) / (ais[i+1] - ais[i])
    else:
        n = (N-1) / 2
        start_i = int((N+1)/2)
        P = (1-p0) / (ee + 2*n - 1)
        p_star = (1 - 2*(n-1)*P - ee*p0 ) / 2

        prob = np.full((len(data), N), P)
        x_lst = np.array(ais) * (ee - 1) * P 
        bins = x_lst[1:-1] * (ee - 1) * P
        digitized = np.digitize(data, bins)

        for i in range(N-1):
            inds = np.argwhere(digitized == i).reshape(-1)
            # x0 < x < x_1
            if i == start_i-1:
                data_inds = data[inds]
                prob[inds, start_i] = (ee*P - p_star) / x_lst[start_i]*data_inds + p_star
                prob[inds, start_i-1] = p0 * (1 - ee) * data_inds / x_lst[start_i] + ee * p0
                prob[inds, start_i-2] = (P - p_star) / x_lst[start_i] * data_inds + p_star

            # -x1 < x < x0
            elif i == start_i-2:
                data_inds = data[inds]
                prob[inds, start_i] = (p_star - P) / x_lst[start_i] * data_inds + p_star
                prob[inds, start_i-1] = p0 * (ee - 1) * data_inds / x_lst[start_i] + ee * p0
                prob[inds, start_i-2] = (p_star - ee*P) / x_lst[start_i]*data_inds + p_star
            
            else:
                # right
                data_inds = data[inds]
                prob[inds, i+1] = (data_inds + (ais[i+1] - ee*ais[i]) * P) / (ais[i+1] - ais[i])
                prob[inds, i] = ((ee*ais[i+1] - ais[i]) * P - data_inds) / (ais[i+1] - ais[i])
                prob[inds, start_i-1] = p0
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    return prob

def construct_type0(eps, N):
    if N % 2 == 0:
        ais = ais_type0(eps, 0, N)
        if len(ais) > 0:
            return ais, 0
        else:
            return [], None
    else:
        n = (N-1) // 2
        if N == 3:
            def Var_1_obj(p0):
                ais = ais_type0(eps, p0[0], N)
                return Var_1(eps ,ais, p0[0], N)
            sol = minimize(Var_1_obj, 1 / (np.exp(eps) + 2*n), bounds=[(0 , 1 / (np.exp(eps) + 2*n))])
            ais = ais_type0(eps, sol.x[0], N)
            return ais, sol.x[0]
        else:
            p0 = 1 / (np.exp(eps) + 2*n)
            ais = ais_type0(eps, p0, N)
            if len(ais) > 0:
                V1 = Var_1(eps, ais, p0, N)
                Vn = Var_n(eps, ais, p0, N)
                if V1 > Vn:
                    return ais, p0
                
                else:
                    ais_0 = ais_type0(eps, 0, N)
                    if Var_1(eps, ais_0, 0, N) < Var_n(eps, ais_0, 0, N):
                        return [], None
                    else:
                        def Var_obj(p0):
                            ais_obj = ais_type0(eps, p0[0], N)
                            return Var_n(eps, ais_obj, p0[0], N) - Var_1(eps, ais_obj, p0[0], N)
                        x0 = 0
                        sol = root(Var_obj, x0)
                        ais = ais_type0(eps, sol.x[0], N)

                        return ais, sol.x[0]
            else:
                return [], None

def construct_type1(eps, N):
    ais = []
    if N % 2 == 0:
        n = N // 2
        p0 = 0
        P = (1-p0) / (np.exp(eps) + 2*n - 1)
        a_n = 1 / (np.exp(eps)- 1) / P
        ais.append(a_n)
        t = (np.exp(eps)-1)*P
        coef_lst = [1/(4*t-1)]
        
    else:
        n = (N-1) // 2
        p0 = 1 / (np.exp(eps) + 2*n)
        P = (1-p0) / (np.exp(eps) + 2*n - 1)
        a_n = 1 / (np.exp(eps)- 1) / P
        ais.append(a_n)
        t = (np.exp(eps)-1)*P
        coef_lst = [1/(4*t-2)]

    for j in range(n-2):
        C = coef_lst[j]
        coef_lst.append((1-2*t + math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C))

    for C in reversed(coef_lst):
        ais.append(ais[-1] * C)

    valid_sequence = check_sequence(ais)
    if valid_sequence:
        ais = make_a(ais, N)
    else:
        ais = []

    return ais, p0
