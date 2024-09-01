import numpy as np
from math import comb

class Duchi():
    def __init__(self, d, eps, rng, range=1.0, k=1):
        self.d = d
        self.eps = eps
        self.rng = rng
        self.range = range
        self.k = k
        self.eps_k = self.eps / self.k
        
        # precomputed
        self.ee = np.exp(self.eps_k)
        self.p = (self.ee-1) / (2*self.ee + 2)
        self.A = (self.ee + 1) / (self.ee - 1)


    def Duchi_single(self, x):
        x = x / self.range
        p = [-x * self.p + 0.5  , x * self.p + 0.5]

        p = np.asarray(p).astype('float64')
        p /= p.sum()
        p = p.reshape(-1)
        
        u = self.rng.choice([-1, 1], 1, p = p)    
        return u * self.A * self.range
    
    def Duchi_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        P1 = data * self.p + 0.5

        u = self.rng.random(len(data))
        minus_idx = np.argwhere(u >= P1).reshape(-1)
        plus_idx = np.argwhere(u < P1).reshape(-1)

        noisy_output[minus_idx] = -self.A
        noisy_output[plus_idx] = self.A

        return noisy_output  * self.range, None
    
    def Duchi_multi_simul_batch(self, data):
        data = data.reshape(-1)
        noisy_output = np.zeros_like(data)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)

        noisy_output[inds], _ = self.Duchi_batch(data[inds])

        return noisy_output * self.d / self.k, None

    def exp_single(self, name, data):
        return self.Duchi_batch(data)
    
    def exp_multi(self, name, data):
        return self.Duchi_multi_simul_batch(data)
