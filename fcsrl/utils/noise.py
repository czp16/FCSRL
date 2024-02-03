import numpy as np

class OUProcess:

    def __init__(self, size, sigma=0.2, 
                 theta=0.15, dt=1e-2, x0=None):
        self.mu = 0.
        self.alpha = theta * dt
        self.beta = sigma * np.sqrt(dt)
        self.size = size # (1,) is batch dim
        self.x0 = x0 if x0 is not None else np.zeros((1,self.size))
        self.reset()

    def sample(self, batch_num):
        self.x = self.x + self.alpha * (self.mu - self.x) + \
            self.beta * np.random.randn(batch_num, self.size)
        return self.x

    def reset(self):
        self.x = self.x0


class GaussianNoise:
    def __init__(self, mu=0.0, sigma=0.1):
        self._mu = mu
        assert 0 <= sigma, "Noise std should not be negative."
        self._sigma = sigma

    def sample(self, size):
        return np.random.normal(self._mu, self._sigma, size)

class ClipGaussianNoise:
    ''' epsilon ~ clip( N(mu, sigma^2), -c, c)

    different from `truncated norm`
    '''
    def __init__(self, mu=0.0, sigma=0.2, c=0.5):
        self._mu = mu
        assert 0 <= sigma, "Noise std should not be negative."
        self._sigma = sigma
        
        if np.isscalar(c):
            c = [-c, c]
        self._c = c

    def sample(self, size):
        s = np.random.normal(self._mu, self._sigma, size)
        s = np.clip(s, self._c[0], self._c[1])
        return s