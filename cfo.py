import numpy as np

class DE():
    def __init__(self, d, eps, bins, rng):
        self.rng = rng
        self.d = d
        self.bins = bins
        self.eps = eps

        # precomputed
        self.ee = np.exp(self.eps)
        self.p = self.ee / (self.ee + self.bins -1)
        self.q = 1 / (self.ee + self.bins - 1)

    def batch(self, data):
        # data is indices of histogram
        data = data.reshape(-1)
        noisy_output = np.zeros_like(data)

        p_random = self.rng.random(len(data))
        p_inds = np.argwhere(p_random < (self.p - self.q)).reshape(-1)
        noisy_output[p_inds] = data[p_inds]
        q_inds = np.argwhere(p_random >= (self.p - self.q)).reshape(-1)
        uniform_random = self.rng.choice([i for i in range(self.bins)], len(q_inds))
        noisy_output[q_inds] = uniform_random

        # noisy output is histogram
        hist = np.bincount(noisy_output, minlength=self.bins)
        est_hist = (hist / len(data) -  self.q) / (self.p - self.q)

        # norm-sub CLS
        while True:
            if np.all(est_hist >= 0):
                break
            est_hist[est_hist < 0] = 0
            non_zero_count = np.count_nonzero(est_hist)
            if non_zero_count > 0:
                delta = (1 - np.sum(est_hist)) / non_zero_count
                est_hist[est_hist > 0] += delta

        return est_hist
    
    def batch_multi_simul(self, data):
        data = data.reshape(-1)
        sample_random = self.rng.random(len(data))
        data = data[sample_random < 1/self.d]

        est_hist = self.batch(data)

        return est_hist

class OUE():
    def __init__(self, d, eps, bins, rng):
        self.rng = rng
        self.d = d
        self.bins = bins
        self.eps = eps

        self.ee = np.exp(self.eps)
        self.p = 0.5
        self.q = 1 / (self.ee + 1)

    def batch(self, data):
        # data is indices of hist (not encoded)
        data = data.reshape(-1)
        # unary encoding
        noisy_output = np.zeros((len(data), self.bins))
        q_random = self.rng.random(noisy_output.shape)
        noisy_output[q_random < self.q] = 1

        row_indices = np.arange(len(data))
        col_indices = data
        noisy_output[row_indices, col_indices] = 0  # Reset the bin for the exact match

        # Randomly select based on p
        p_random = self.rng.random(len(data))
        p_mask = p_random < self.p
        noisy_output[row_indices[p_mask], col_indices[p_mask]] = 1

        support_sum = noisy_output.sum(axis=0)
        est_hist = (support_sum / len(data) - self.q) / (self.p - self.q)

        # norm-sub CLS
        while True:
            if np.all(est_hist >= 0):
                break
            est_hist[est_hist < 0] = 0
            non_zero_count = np.count_nonzero(est_hist)
            if non_zero_count > 0:
                delta = (1 - np.sum(est_hist)) / non_zero_count
                est_hist[est_hist > 0] += delta
        
        return est_hist
    
    def batch_multi_simul(self, data):
        data = data.reshape(-1)
        sample_random = self.rng.random(len(data))
        data = data[sample_random < 1/self.d]

        est_hist = self.batch(data)

        return est_hist
    
