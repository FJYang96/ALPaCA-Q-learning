################################################################
# Fengjun Yang, 2018
# This is a class that wraps data normalization functionalities
################################################################

import numpy as np

class mmcNorm:
    """Normalize and denormalize data based on the empirical distribution"""

    def __init__(self, config):
        """ Read the empirical mean and variance from the config file """
        self.pos_mean = np.mean(config['pos_range'])
        self.pos_var = config['pos_var']
        self.vel_mean = np.mean(config['vel_range'])
        self.vel_var = config['vel_var']
        self.q_mean  = config['q_mean']
        self.q_var   = config['q_var']

    def norm_x(self, x):
        x_norm = np.zeros(x.shape)
        x_norm[:,:,0] = (x[:,:,0] - self.pos_mean) / self.pos_var 
        x_norm[:,:,1] = (x[:,:,1] - self.vel_mean) / self.vel_var
        return x_norm

    def norm_y(self, y):
        y_norm = (y - self.q_mean) / self.q_var
        return y_norm

    def denorm_x(self, x_norm):
        x = np.zeros(x_norm.shape)
        x[:,:,0] = x_norm[:,:,0] * self.pos_var + self.pos_mean
        x[:,:,1] = x_norm[:,:,1] * self.vel_var + self.vel_mean
        return x

    def denorm_y(self, y_norm):
        y = y_norm * self.q_var + self.q_mean
        return y

    def denorm_var_pred(self, y_var):
        return y_var * (self.q_var ** 2)
