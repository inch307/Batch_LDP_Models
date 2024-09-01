import numpy as np

class TOPM():
    def __init__(self, d, eps, rng, range=1.0, k=1, t=None):
        self.d = d
        self.eps = eps
        self.rng = rng
        self.range = range
        # eps_star for TO_single
        self.k = k
        self.eps_k = self.eps / self.k
        if t is None:
            self.t = np.exp(self.eps_k / 3)
        else:
            self.t = t
        self.P0_0, self.C = self.get_Ps()

        self.Px_0, self.Px_1 = self.construct_prob()

        # for piecewise mech
        self.A = (np.exp(self.eps_k) + self.t) * (self.t + 1) / (self.t * (np.exp(self.eps_k)-1))

        # HM
        self.beta = self.get_beta()
        if self.eps_k < np.log(2):
            self.x_star = 0
        else:
            self.x_star = (self.beta - 1) * self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2 / ( 2 * (np.exp(self.eps_k) - self.P0_0)**2 * (self.beta * (np.exp(self.eps_k) + self.t) - np.exp(self.eps_k) + 1))

    def construct_prob(self):
        Px_1 = (np.exp(self.eps_k) - self.P0_0) / (np.exp(self.eps_k) + 1) - (1 - self.P0_0) / 2
        Px_0 = self.P0_0 / np.exp(self.eps_k) - self.P0_0

        return Px_0, Px_1

    def get_Ps(self):
        if self.eps_k < np.log(2):
            P0_0 = 0
        elif self.eps_k <= np.log((3 + np.sqrt(65)) / 2):
            delta_0 = np.exp(4*self.eps_k) + 14*np.exp(3*self.eps_k) + 50*np.exp(2*self.eps_k) - 2 * np.exp(self.eps_k) + 25
            delta_1 = -2*np.exp(6*self.eps_k) -42*np.exp(5*self.eps_k) -270*np.exp(4*self.eps_k) -404*np.exp(3*self.eps_k) -918*np.exp(2*self.eps_k) +30*np.exp(self.eps_k) -250
            
            P0_0 = (-1/6) * ( -np.exp(2*self.eps_k) - 4*np.exp(self.eps_k) - 5 + 2 * np.sqrt(delta_0) * np.cos(np.pi/3 + (1/3) * np.arccos(-delta_1 / (2*np.sqrt(delta_0**3))) ))
        else:
            P0_0 = np.exp(self.eps_k) / (np.exp(self.eps_k) + 2)

        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k)-1) * (1 - P0_0 / np.exp(self.eps_k)))

        return P0_0, C

    def get_beta(self):
        if self.eps_k > 0 and self.eps_k < 0.610986:
            beta = 0
        elif self.eps_k < np.log(2):
            beta = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)
        else:
            c = np.exp(self.eps_k)
            A = self.P0_0**2 * c**2 * (c+1)**4 / (4 * (c + self.t)**2 * (c - self.P0_0)**4 * (c-1)) - self.P0_0**2 * c**2 * (c+1)**4 / (2 * (c+self.t)**2 * (c-1) * (c-self.P0_0)**4) + ( (self.t+1)**3 + c - 1) / (3 * self.t**2 * (c - 1)**2) - (1 - self.P0_0) * c**2 * (c+1)**2 / ((c+self.t)*(c-1)**2 * (c-self.P0_0)**2)
            B = (-(1 + self.t)**2 * self.P0_0**2 * c**2 * (c+1)**4) / (4* (c+self.t)**2 * (c-self.P0_0)**4 * (c-1))

            beta1 = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)

            beta = (-np.sqrt(B/A) + np.exp(self.eps_k) -1) / (np.exp(self.eps_k) + np.exp(self.eps_k/3))
        
        return beta

    def TO_single(self, x):
        x = x / self.range
        P_0_x = self.P0_0 + self.Px_0 * np.abs(x)

        if x >= 0 and x <= 1:
            P_C_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_mC_x = 1 - P_C_x - P_0_x
        else:
            P_mC_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_C_x = 1 - P_0_x - P_mC_x

        p = [P_mC_x, P_0_x, P_C_x]
        p = np.asarray(p).astype('float64')
        p /= p.sum()
        u = self.rng.choice([-1, 0, 1], 1, p = p)

        return u * self.C * self.range
    
    def TO_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)
        
        minus_idx = np.argwhere(data < 0).reshape(-1)
        P_0_x = self.P0_0 + self.Px_0 * np.abs(data[minus_idx])
        P_mC_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(data[minus_idx])

        minus_random = self.rng.random(len(minus_idx))
        mC_idx = np.argwhere(minus_random < P_mC_x).reshape(-1)
        zero_idx = np.argwhere((minus_random >= P_mC_x) & (minus_random < (P_mC_x + P_0_x)) ).reshape(-1)
        C_idx = np.argwhere(minus_random >= (P_mC_x + P_0_x)).reshape(-1)
        noisy_output[minus_idx[mC_idx]] = -self.C
        noisy_output[minus_idx[zero_idx]] = 0
        noisy_output[minus_idx[C_idx]] = self.C

        plus_idx = np.argwhere(data >= 0).reshape(-1)
        P_0_x = self.P0_0 + self.Px_0 * np.abs(data[plus_idx])
        P_C_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(data[plus_idx])

        plus_random = self.rng.random(len(plus_idx))
        C_idx = np.argwhere(plus_random < P_C_x).reshape(-1)
        zero_idx = np.argwhere((plus_random >= P_C_x) & (plus_random < (P_C_x + P_0_x)) ).reshape(-1)
        mC_idx = np.argwhere(plus_random >= (P_C_x + P_0_x)).reshape(-1)
        noisy_output[plus_idx[mC_idx]] = -self.C
        noisy_output[plus_idx[zero_idx]] = 0
        noisy_output[plus_idx[C_idx]] = self.C

        return noisy_output * self.range, None
    
    def TO_max_var(self):
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))

        if self.C**2*b < 2:
            V = (1 - self.P0_0) * self.C**2 + self.C**4 * b**2 / 4
        else:
            V = (1 - self.P0_0 + b) * self.C**2 - 1
        return V
    
    def TO_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.TO_single(x[j]) / self.k

        return y.reshape(original_shape)
    
    def TO_multi_batch(self, data):
        sampled_dims = np.random.choice(len(data), self.k, replace=False)
        noisy_output = np.zeros_like(data)

        noisy_output[sampled_dims], _ = self.TO_batch(data[sampled_dims])
        return noisy_output * self.d / self.k
    
    def TO_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.TO_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def TO_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds], _ = self.TO_batch(data[inds])

        return noisy_output * self.d / self.k, None
    
    def TO_var(self, x):
        b = self.P0_0*(1 - 1/np.exp(self.eps_k))
        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k) -1) * (1 - self.P0_0 / np.exp(self.eps_k)))

        return C**2 * (1 - self.P0_0) + C**2 * b * np.abs(x) - x**2
    
    def PM_single(self, x):
        u = self.rng.random()
        x = x / self.range

        l = (np.exp(self.eps_k) + self.t) * (x * self.t - 1) / (self.t * (np.exp(self.eps_k) - 1))
        r = (np.exp(self.eps_k) + self.t) * (x * self.t + 1) / (self.t * (np.exp(self.eps_k) - 1))

        if u < np.exp(self.eps_k) / (self.t + np.exp(self.eps_k)):
            y = self.rng.uniform(l, r)
        else:
            length_l = abs(l - (-self.A))
            legnth_r = abs(self.A - r)
            interval_l = length_l / (length_l + legnth_r)
            
            if self.rng.random() < interval_l:
                y = self.rng.uniform(-self.A, l)
            else:
                y = self.rng.uniform(r, self.A)

        return y * self.range
    
    def PM_multi(self, x, m=None):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])

        for j in js:
            noisy_output = self.PM_single(x[j])
            if m is not None:
                noisy_output = utils.discretization(self.A, noisy_output / self.range, self.rng, m) * self.range
            y[j] = self.d * noisy_output / self.k

        return y.reshape(original_shape)
    
    def PM_multi_batch(self, data):
        sampled_dims = np.random.choice(len(data), self.k, replace=False)
        noisy_output = np.zeros_like(data)

        noisy_output[sampled_dims], _ = self.PM_batch(data[sampled_dims])
        return noisy_output * self.d / self.k
    
    def PM_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.PM_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def PM_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds], _ = self.PM_batch(data[inds])

        return noisy_output * self.d / self.k, inds
    
    def PM_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        u = self.rng.random(len(noisy_output))

        l = (np.exp(self.eps_k) + self.t) * (data * self.t - 1) / (self.t * (np.exp(self.eps_k) - 1))
        r = (np.exp(self.eps_k) + self.t) * (data * self.t + 1) / (self.t * (np.exp(self.eps_k) - 1))

        inner_idx = np.argwhere(u < np.exp(self.eps_k) / (self.t + np.exp(self.eps_k))).reshape(-1)
        outer_idx = np.argwhere(u >= np.exp(self.eps_k) / (self.t + np.exp(self.eps_k))).reshape(-1)

        inner_y = self.rng.random(len(inner_idx))
        inner_y = (r[inner_idx]-l[inner_idx])*inner_y + l[inner_idx]
        noisy_output[inner_idx] = inner_y

        length_l = np.abs(l[outer_idx] + self.A)
        legnth_r = np.abs(self.A - r[outer_idx])
        interval_l = length_l / (length_l + legnth_r)
        interval_random = self.rng.random(len(outer_idx))
        left_idx = outer_idx[interval_random < interval_l]
        right_idx = outer_idx[interval_random >= interval_l]
        
        left_y = self.rng.random(len(left_idx))
        left_y = (l[left_idx] + self.A) * left_y - self.A
        noisy_output[left_idx] = left_y

        right_y = self.rng.random(len(right_idx))
        right_y = (self.A - r[right_idx]) * right_y + r[right_idx]
        noisy_output[right_idx] = right_y

        return noisy_output * self.range, None

    def HM_single(self, x):
        if self.rng.random() > self.beta:
            y = self.TO_single(x)
        else:
            y = self.PM_single(x)

        return y
    
    def HM_multi_batch(self, data):
        sampled_dims = np.random.choice(len(data), self.k, replace=False)
        noisy_output = np.zeros_like(data)

        noisy_output[sampled_dims], _ = self.HM_batch(data[sampled_dims])
        return noisy_output * self.d / self.k
    
    def HM_batch(self, data):
        data = data.reshape(-1)
        hm_random = self.rng.random(len(data))
        to_inds = np.argwhere(hm_random > self.beta).reshape(-1)
        pm_inds = np.argwhere(hm_random <= self.beta).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[to_inds], _ = self.TO_batch(data[to_inds])
        noisy_output[pm_inds], _ = self.PM_batch(data[pm_inds])

        return noisy_output, [to_inds, pm_inds]
    
    def HM_var(self, x):
        a = self.P0_0
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))

        return self.beta * ((self.t+1) * x**2 / (np.exp(self.eps_k)-1) + (self.t +np.exp(self.eps_k)) * (((self.t+1)**3) + np.exp(self.eps_k) - 1) / (3*self.t**2 * (np.exp(self.eps_k) -1)**2)) + (1 - self.beta) *( ( (1 - a) * np.exp(2*self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ((np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k)-a)**2)) + b * np.abs(x) * np.exp(2 * self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ( (np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k) - a) **2 ) - x**2 )

    def HM_max_var(self):
        b = self.beta
        v = self.HM_var(self.x_star)
        V0 = self.HM_var(0)
        V1 = self.HM_var(1)
        V = max(V0, V1)
        V = max(V, v)

        return V

    def HM_multi(self, x, m=None):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            noisy_output = self.HM_single(x[j])
            y[j] = self.d * noisy_output / self.k

        return y.reshape(original_shape)
    
    def HM_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.HM_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def HM_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        noisy_output = np.zeros_like(data)

        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output[inds], to_pm_inds = self.HM_batch(data[inds])
        to_inds = inds[to_pm_inds[0]]
        pm_inds = inds[to_pm_inds[1]]
        return noisy_output * self.d / self.k, [to_inds, pm_inds]
    
    
    def exp_batch(self, name, data):
        if name == 'to':
            return self.TO_batch(data)
        elif name =='topm':
            return self.HM_batch(data)
        elif name == 'pm_sub':
            return self.PM_batch(data)
    
    def exp_multi(self, name, data):
        if name == 'to':
            return self.TO_multi_simul_batch(data)
        elif name == 'topm':
            return self.HM_multi_simul_batch(data)
        elif name == 'pm_sub':
            return self.PM_multi_simul_batch(data)
        
    def multi(self, name, data):
        if name == 'to':
            return self.TO_multi_batch(data)
        elif name == 'topm':
            return self.HM_multi_batch(data)
        elif name == 'pm_sub':
            return self.PM_multi_batch(data)
