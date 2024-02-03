import numpy as np

# Running mean std from lib 'baselines'
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class BaseNormalizer:
    def __init__(self, read_only=True):
        self.read_only = read_only

    def __call__(self, x):
        return x

    def normalize(self, x):
        return self(x)

    def unnormalize(self, x):
        return x
    
    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=True, clip=50.0, epsilon=1e-20):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def unnormalize(self, x):
        x = np.asarray(x)
        return x * np.sqrt(self.rms.var + self.epsilon) + self.rms.mean

    def state_dict(self):
        return {
            'mean': self.rms.mean,
            'var': self.rms.var,
            'cnt': self.rms.count,
        }

    def load_state_dict(self, saved):
        if self.rms is None:
            self.rms = RunningMeanStd(shape=saved['mean'].shape)
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']
        self.rms.count = saved['cnt']
