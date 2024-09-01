import numpy as np

def EM(prob, d, noisy_hist, tau=1e-3, smoothing=False):
    est_hist = np.ones(d) / d
    E = np.zeros_like(est_hist)
    noisy_d = len(noisy_hist)

    # E step
    while(1):
        sum_log = np.log(np.dot(prob, est_hist))
        old_loglikelyhood = np.dot(noisy_hist, sum_log)

        pr_i = np.dot(prob, est_hist)
        E = est_hist * (np.dot(noisy_hist / pr_i, prob))

        # M step
        nom = np.sum(E)
        est_hist = E / nom

        sum_log = np.log(np.dot(prob, est_hist))
        loglikelyhood = np.dot(noisy_hist, sum_log)
        

        if smoothing:
            est_hist[1:-1] = est_hist[1:-1] / 2 + (est_hist[:-2] + est_hist[2:])/4

        if np.abs(loglikelyhood - old_loglikelyhood) < tau:
            break
    return est_hist

def EM_MAP(prob, d, noisy_hist, mu, squared_sigma, tau=1e-3, smoothing=False):
    est_hist = mu.copy()
    # est_hist = np.ones(d) / d
    E = np.zeros_like(est_hist)
    noisy_d = len(noisy_hist)

    # E step
    while(1):
        sum_log = np.log(np.dot(prob, est_hist))
        old_loglikelyhood = np.dot(noisy_hist, sum_log)
        
        pr_i = np.dot(prob, est_hist)
        E = est_hist * (np.dot(noisy_hist / pr_i, prob))

        # M step
        nom = np.sum(E)
        est_hist = (E + (1/squared_sigma) * mu) / (nom + (1/squared_sigma))
        
        sum_log = np.zeros(noisy_d)
        sum_log = np.log(np.dot(prob, est_hist))
        loglikelyhood = np.inner(noisy_hist, sum_log)

        if smoothing:
            est_hist[1:-1] = est_hist[1:-1] / 2 + (est_hist[:-2] + est_hist[2:])/4

        if np.abs(loglikelyhood - old_loglikelyhood) < tau:
            break
    return est_hist

def get_bin_idx(bin, x):
    for i in range(len(bin)):
        if x < bin[i]:
            return i
    return len(bin)-1

def plus_L(x, a_l, a_r, P, ee):
    return ((ee-1)*a_r*P*x - 0.5*x**2) / (a_r - a_l)

def plus_R(x, a_l, a_r, P, ee):
    return ((x**2 / 2) - a_l*(ee - 1)*P*x) / (a_r - a_l)

def minus_L(x, a_l, a_r, P, ee):
    return (0.5*x**2 - (ee-1)*a_r*P*x) / (a_l - a_r)

def minus_R(x, a_l, a_r, P, ee):
    return (a_l*(ee-1)*P*x - (x**2 / 2)) / (a_l - a_r)

class ProbNM():
    def __init__(self, d, no_mech):
        self.d = d
        self.N = no_mech.best_hybrid_mech['N']
        self.ais = no_mech.best_hybrid_mech['ais']
        self.p0 = no_mech.best_hybrid_mech['p0']
        self.eps = no_mech.eps_k

        if self.N % 2 == 0:
            n = self.N / 2
            start_i = int(self.N/2)
        else:
            n = (self.N-1) / 2
            start_i = int((self.N+1)/2)

        # self.prob = np.full((self.N, self.d), self.P)
        self.prob = np.zeros((self.N, self.d))
        self.ee = np.exp(self.eps)
        self.P = (1-self.p0) / (self.ee + 2*n - 1)
        self.p_star = (1 - 2*(n-1)*self.P - self.ee*self.p0 ) / 2
        self.t = (self.ee - 1) * self.P

        # precomputed terms

        # data histogram
        self.l = 2 / self.d
        self.bin = []
        for i in range(self.d):
            self.bin.append(-1 + (i+1) * self.l)
        # print(self.bin)
        
        # self.ais_transformed = self.ais * (np.exp(self.eps)  - 1) * self.P

        if self.N % 2 == 0:
            for i in range(self.d):
                for j in range(self.N):
                    left = self.bin[i] - self.l
                    right = self.bin[i]
                    prob = self.P
                    if j == 0:
                        ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                        u2 = min(self.t * self.ais[j+1], right)
                        d2 = max(self.t * self.ais[j], left)
                        prob += ind_2 *  ( self.t*self.ais[j+1]*(u2 - d2) - 0.5 * (u2**2 - d2**2)   ) / (self.l * (self.ais[j+1] - self.ais[j]))
                    elif j == (self.N - 1):
                        ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                        u1 = min(self.t * self.ais[j], right)
                        d1 = max(self.t * self.ais[j-1], left)
                        prob += ind_1 * (0.5*(u1**2 - d1**2) - self.t*self.ais[j-1]*(u1 - d1)  ) / (self.l * (self.ais[j] - self.ais[j-1]))
                    else:
                        ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                        u1 = min(self.t * self.ais[j], right)
                        d1 = max(self.t * self.ais[j-1], left)
                        prob += ind_1 * (0.5*(u1**2 - d1**2) - self.t*self.ais[j-1]*(u1 - d1)  ) / (self.l * (self.ais[j] - self.ais[j-1]))
                        ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                        u2 = min(self.t * self.ais[j+1], right)
                        d2 = max(self.t * self.ais[j], left)
                        prob += ind_2 *  ( self.t*self.ais[j+1]*(u2 - d2) - 0.5 * (u2**2 - d2**2)   ) / (self.l * (self.ais[j+1] - self.ais[j])) 
                    self.prob[j][i] = prob

        else:
            for i in range(self.d):
                for j in range(self.N):
                    left = self.bin[i] - self.l
                    right = self.bin[i]
                    
                    # a1
                    if j == start_i:
                        prob = self.P
                        ind_2 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                        ind_3 = int(left <= self.t * self.ais[j-1] and right >= self.t * self.ais[j-2])

                        if self.N > 3:
                            ind_1 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                            u1 = min(self.t * self.ais[j+1], right)
                            d1 = max(self.t * self.ais[j], left)
                            prob += ind_1 * (self.t*self.ais[j+1]*(u1 - d1) - 0.5*(u1**2 - d1**2)) / (self.l * (self.ais[j+1] - self.ais[j]))

                        u2 = min(self.t * self.ais[j], right)
                        d2 = max(self.t * self.ais[j-1], left)
                        prob += ind_2 * ( ( 0.5 * (u2**2 - d2**2) * (self.ee * self.P - self.p_star) ) / (self.t * self.ais[j]) + (self.p_star - self.P) * (u2 - d2) ) / self.l

                        u3 = min(self.t * self.ais[j-1], right)
                        d3 = max(self.t * self.ais[j-2], left)
                        prob += ind_3 *  ( ( 0.5 * (u3**2 - d3**2) * (self.p_star - self.P) ) / (self.t * self.ais[j]) + (self.p_star - self.P) * (u3 - d3) ) / self.l

                    # a0
                    elif j == start_i-1:
                        prob = self.p0
                        ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                        u1 = min(self.t * self.ais[j], right)
                        d1 = max(self.t * self.ais[j-1], left)
                        prob += ind_1 * ( ( self.ee - 1 ) * self.p0 * 0.5 * (u1**2 - d1**2) / (self.t * self.ais[j+1]) + (self.ee -1)*self.p0*(u1 - d1) ) / self.l

                        ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                        u2 = min(self.t * self.ais[j+1], right)
                        d2 = max(self.t * self.ais[j], left)
                        prob += ind_2 * ( ( 1 - self.ee ) * self.p0 * 0.5 * (u2**2 - d2**2) / (self.t * self.ais[j+1]) + (self.ee -1)*self.p0*(u2 - d2) ) / self.l

                    # -a1
                    elif j == start_i-2:
                        prob = self.P
                        ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                        ind_3 = int(left <= self.t * self.ais[j+2] and right >= self.t * self.ais[j+1])

                        if self.N > 3:
                            ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                            u1 = min(self.t * self.ais[j], right)
                            d1 = max(self.t * self.ais[j-1], left)
                            prob += ind_1 * (0.5*(u1**2 - d1**2) - self.t*self.ais[j-1]*(u1 - d1)) / (self.l * (self.ais[j] - self.ais[j-1]))

                        u2 = min(self.t * self.ais[j+1], right)
                        d2 = max(self.t * self.ais[j], left)
                        prob += ind_2 * ( ( 0.5 * (u2**2 - d2**2) * (self.p_star - self.ee * self.P) ) / (self.t * -self.ais[j]) + (self.p_star - self.P) * (u2 - d2) ) / self.l

                        u3 = min(self.t * self.ais[j+2], right)
                        d3 = max(self.t * self.ais[j+1], left)
                        prob += ind_3 *  ( ( 0.5 * (u3**2 - d3**2) * (self.P - self.p_star) ) / (self.t * -self.ais[j]) + (self.p_star - self.P) * (u3 - d3) ) / self.l
                    else:
                        prob = self.P
                        if j == 0:
                            ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                            u2 = min(self.t * self.ais[j+1], right)
                            d2 = max(self.t * self.ais[j], left)
                            prob += ind_2 *  ( self.t*self.ais[j+1]*(u2 - d2) - 0.5 * (u2**2 - d2**2)   ) / (self.l * (self.ais[j+1] - self.ais[j]))
                        elif j == (self.N - 1):
                            ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                            u1 = min(self.t * self.ais[j], right)
                            d1 = max(self.t * self.ais[j-1], left)
                            prob += ind_1 * (0.5*(u1**2 - d1**2) - self.t*self.ais[j-1]*(u1 - d1)  ) / (self.l * (self.ais[j] - self.ais[j-1]))
                        else:
                            ind_1 = int(left <= self.t * self.ais[j] and right >= self.t * self.ais[j-1])
                            u1 = min(self.t * self.ais[j], right)
                            d1 = max(self.t * self.ais[j-1], left)
                            prob += ind_1 * (0.5*(u1**2 - d1**2) - self.t*self.ais[j-1]*(u1 - d1)  ) / (self.l * (self.ais[j] - self.ais[j-1]))
                            ind_2 = int(left <= self.t * self.ais[j+1] and right >= self.t * self.ais[j])
                            u2 = min(self.t * self.ais[j+1], right)
                            d2 = max(self.t * self.ais[j], left)
                            prob += ind_2 *  ( self.t*self.ais[j+1]*(u2 - d2) - 0.5 * (u2**2 - d2**2)   ) / (self.l * (self.ais[j+1] - self.ais[j])) 
                    
                    self.prob[j][i] = prob

class ProbPM_LR():
    def __init__(self, d, k, eps, noisy_d = None):
        self.d = d
        self.k = k
        self.eps = eps / k
        self.t = np.exp(self.eps / 3)
        self.ee = np.exp(self.eps)

        # precomputed term
        self.PM_term = (self.ee + self.t) / self.t / (self.ee - 1)
        self.length_PM = 2 * self.PM_term

        self.A = self.PM_term * (self.t + 1)
        self.low_p = self.t * (self.ee - 1) / 2 / (self.t + self.ee)**2
        self.high_p = self.ee * self.low_p
        self.h = self.high_p - self.low_p

        # data histogram
        self.l = 2 / self.d
        self.bin = -1 + (np.arange(1, self.d + 1) * self.l)

        # output histogram
        self.noisy_len_bin = 2 * (self.ee + self.t) / (self.ee -1) / self.d
        self.noisy_d = int(np.ceil((2 * self.A) // self.noisy_len_bin))
        # print(f'eps: {eps}')
        # print(self.noisy_d)
        self.l_ = 2 * self.A / self.noisy_d
        self.bin_ = -self.A + (np.arange(1, self.noisy_d + 1) * self.l_)

        self.prob = np.zeros((self.noisy_d, self.d))

        self.compute_probs()

    def L(self, x):
        return self.PM_term * (x * self.t - 1)
    
    def R(self, x):
        return self.PM_term * (x * self.t + 1)
    
    def Lp(self, a):
        return (a * self.t * (self.ee - 1) + self.ee + self.t) / self.t / (self.ee + self.t)
    
    def Rp(self, a):
        return (a * self.t * (self.ee - 1) - (self.ee + self.t)) / self.t / (self.ee + self.t)
    
    def integral_L(self, u, d):
        return self.PM_term * ((0.5 * self.t * u**2 - u) - (0.5 * self.t * d**2 - d))
    
    def integral_R(self, u, d):
        return self.PM_term * ((0.5 * self.t * u**2 + u) - (0.5 * self.t * d**2 + d))

    def compute_prob_ij(self, left, right, left_, right_):
        prob = self.low_p * self.l_ * self.l

        if self.length_PM >= self.l_:
            ind_1 = (self.R(left) <= right_) & (self.R(right) >= left_)
            u1 = np.minimum(self.Rp(right_), right)
            d1 = np.maximum(self.Rp(left_), left)
            ind_2 = (self.L(left) <= left_) & (self.R(right) >= right_)
            u2 = np.minimum(self.Lp(left_), right)
            d2 = np.maximum(self.Rp(right_), left)
            ind_3 = (self.L(left) <= right_) & (self.L(right) >= left_)
            u3 = np.minimum(self.Lp(right_), right)
            d3 = np.maximum(self.Lp(left_), left)
            prob += ind_1 * (self.integral_R(u1, d1) - (u1 * left_ - d1 * left_)) * self.h
            prob += ind_2 * (self.l_ * self.h * (u2 - d2))
            prob += ind_3 * (right_ * (u3 - d3) - self.integral_L(u3, d3)) * self.h
        else:
            ind_1 = (self.L(left) <= left_) & (self.R(right) >= left_)
            u1 = np.minimum(self.Lp(left_), right)
            d1 = np.maximum(self.Rp(left_), left)
            ind_2 = (self.R(left) <= right_) & (self.L(right) >= left_)
            u2 = np.minimum(self.Rp(right_), right)
            d2 = np.maximum(self.Lp(left_), left)
            ind_3 = (self.L(left) <= right_) & (self.R(right) >= right_)
            u3 = np.minimum(self.Lp(right_), right)
            d3 = np.maximum(self.Rp(right_), left)
            prob += ind_1 * (self.integral_R(u1, d1) - (u1 * left_ - d1 * left_)) * self.h
            prob += ind_2 * (self.length_PM * self.h * (u2 - d2))
            prob += ind_3 * (right_ * (u3 - d3) - self.integral_L(u3, d3)) * self.h

        return prob

    def compute_probs(self):
        left_points = self.bin - self.l
        right_points = self.bin
        left_points_ = self.bin_ - self.l_
        right_points_ = self.bin_

        for i in range(self.noisy_d):
            self.prob[i, :] = self.compute_prob_ij(left_points, right_points, left_points_[i], right_points_[i]) / self.l
