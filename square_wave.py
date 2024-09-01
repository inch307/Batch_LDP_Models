import numpy as np

class SW():
    def __init__(self, d, eps, rng):
        # assume range [0, 1]
        self.d = d
        self.eps = eps
        self.rng = rng
        
        self.ee = np.exp(eps)
        self.b = (self.eps * self.ee - self.ee + 1) / (2*self.ee * (self.ee -1 -self.eps))
        self.q = 1 / (2 *self.ee*self.b + 1)
        self.p = self.q * self.ee

    def SW_single(self, x):
        l = x - self.b
        r = x + self.b
        if self.rng.random() < self.p:
            y = self.rng.uniform(l, r)
        else:
            length_l = abs(l - (-self.b))
            length_r = abs(1+self.b - r)
            interval_l = length_l / (length_l + length_r)

            if self.rng.random() < interval_l:
                y = self.rng.uniform(-self.b, l)
            else:
                y = self.rng.uniform(r, 1 + self.b)

        return y
    
    def SW_batch(self, data):
        data = data.reshape(-1)
        l = data - self.b
        r = data + self.b
        noisy_output = np.zeros_like(data)

        p_random = self.rng.random(len(data))
        inner_inds = np.argwhere(p_random < self.p).reshape(-1)
        lr_random = self.rng.random(len(inner_inds))
        noisy_output[inner_inds] = lr_random * (r - l) + l

        outer_inds = np.argwhere(p_random >= self.p).reshape(-1)
        length_l = np.abs(l[outer_inds] + self.b)
        length_r = np.abs(1 + self.b - r[outer_inds])
        interval_l = length_l / (length_l + length_r)
        interval_random = self.rng.random(len(outer_inds))
        left_inds = outer_inds[interval_random < interval_l]
        right_inds = outer_inds[interval_random >= interval_l]

        left_y = self.rng.random(len(left_inds))
        left_y = (l[left_inds] + self.b) * left_y - self.b
        noisy_output[left_inds] = left_y

        right_y = self.rng.random(len(right_inds))
        right_y = (1 +self.b -r[right_inds]) * right_y + r[right_inds]
        noisy_output[right_inds] = right_y

        return noisy_output
    
    def SW_batch_multi_simul(self, data):
        data = data.reshape(-1)
        inds = np.argwhere(self.rng.random(len(data)) < 1/self.d).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds] = self.SW_batch(data[inds])

        return noisy_output, inds

    def get_prob(self, d, noisy_d):
        # data histogram
        l = 1 / d
        bin = (np.arange(1, d + 1) * l)

        # output histogram
        l_ = (1 + 2 * self.b) / noisy_d
        bin_ = (-self.b + (np.arange(1, noisy_d + 1) * l_))

        h = self.p - self.q
        length_SW = 2 * self.b

        left_points = bin - l
        right_points = bin
        left_points_ = bin_ - l_
        right_points_ = bin_

        prob = np.zeros((noisy_d, d))

        for i in range(noisy_d):
            left_ = left_points_[i]
            right_ = right_points_[i]

            prob[i, :] = self.q * l_ * l

            if length_SW >= l_:
                ind_1 = (self.R(left_points) <= right_) & (self.R(right_points) >= left_)
                u1 = np.minimum(self.Rp(right_), right_points)
                d1 = np.maximum(self.Rp(left_), left_points)
                ind_2 = (self.L(left_points) <= left_) & (self.R(right_points) >= right_)
                u2 = np.minimum(self.Lp(left_), right_points)
                d2 = np.maximum(self.Rp(right_), left_points)
                ind_3 = (self.L(left_points) <= right_) & (self.L(right_points) >= left_)
                u3 = np.minimum(self.Lp(right_), right_points)
                d3 = np.maximum(self.Lp(left_), left_points)

                prob[i, :] += ind_1 * (self.integral_R(u1, d1) - (u1 * left_ - d1 * left_)) * h
                prob[i, :] += ind_2 * (l_ * h * (u2 - d2))
                prob[i, :] += ind_3 * (right_ * (u3 - d3) - self.integral_L(u3, d3)) * h
            else:
                ind_1 = (self.L(left_points) <= left_) & (self.R(right_points) >= left_)
                u1 = np.minimum(self.Lp(left_), right_points)
                d1 = np.maximum(self.Rp(left_), left_points)
                ind_2 = (self.R(left_points) <= right_) & (self.L(right_points) >= left_)
                u2 = np.minimum(self.Rp(right_), right_points)
                d2 = np.maximum(self.Lp(left_), left_points)
                ind_3 = (self.L(left_points) <= right_) & (self.R(right_points) >= right_)
                u3 = np.minimum(self.Lp(right_), right_points)
                d3 = np.maximum(self.Rp(right_), left_points)

                prob[i, :] += ind_1 * (self.integral_R(u1, d1) - (u1 * left_ - d1 * left_)) * h
                prob[i, :] += ind_2 * (length_SW * h * (u2 - d2))
                prob[i, :] += ind_3 * (right_ * (u3 - d3) - self.integral_L(u3, d3)) * h

            prob[i, :] /= l

        return prob

    
    def Lp(self, a):
        return self.b + a
    
    def Rp(self,a):
        return a - self.b
    
    def L(self, x):
        return x - self.b
    
    def R(self, x):
        return x + self.b
    
    def integral_L(self, u, d):
        return 0.5 * (u**2 - d**2) - self.b* (u - d)
    
    def integral_R(self, u, d):
        return 0.5 * (u**2 - d**2) + self.b * (u-d)