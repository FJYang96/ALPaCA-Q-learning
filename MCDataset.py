from main.dataset import Dataset, PresampledTrajectoryDataset
from VIAgent import MCVI
import numpy as np
import yaml

class MCOnlineSampleDataset(Dataset):
    """
    Samples a new trajectory from MountainCarTransfer environment every
    time a new data trajectory is requested
    """
    def __init__(self, config, noise_var=None, rng=None):
        """
        Specify the data generation parameters
        """
        self.gran = config['gran'] 
        self.step_limit = config['step_limit']
        self.pos_range = config['pos_range']
        self.vel_range = config['vel_range']
        self.thrust_range = config['thrust_range']
        self.gravity_range = config['gravity_range']
        # Assume noise is known (as in ALPaCA)
        if noise_var is None:
            self.noise_std = np.sqrt( config['sigma_eps'] )
        else:
            self.noise_std = np.sqrt( noise_var )
        # Can specify the rng if desired
        self.np_random = rng
        if rng is None:
            self.np_random = np.random

    def sample_parameter(self, param_range, num):
        """
        Given the lower and upper bound of parameter, sample num samples
        param_range should be a tuple/list: [lower, upper]
        """
        param = param_range[0] \
            + self.np_random.random(num) * (param_range[1] - param_range[0])
        return param

    def sample(self, n_funcs, n_samples, return_lists=False):
        """
        Sample n_funcs new trajectories, each with n_samples
        """
        x_dim = 2
        y_dim = 3
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        grav = self.sample_parameter(self.gravity_range, n_funcs)
        thru = self.sample_parameter(self.thrust_range, n_funcs)
        for i in range(n_funcs):
            VIagent = MCVI(self.gran, self.step_limit, \
                           gravity=grav[i], thrust=thru[i])
            VIagent.value_iteration()
            pos_samp = self.sample_parameter(self.pos_range, n_samples)
            vel_samp = self.sample_parameter(self.vel_range, n_samples)
            y_samp = VIagent.get_values(pos_samp, vel_samp)\
                + self.noise_std * self.np_random.randn(n_samples, 3)
            x_samp = np.stack([pos_samp, vel_samp]).T
            x[i,:,:] = x_samp
            y[i,:,:] = y_samp

        if return_lists:
            raise NotImplementedError

        return x, y

class MCOfflineDataset(Dataset):
    """
    Reads presampled q-value data from directory as a wrapper for ALPaCA
    training
    """
    def __init__(self, config):
        self.dir = config['data_dir']
        self.num_data = config['num_data']
        self.x_dim = config['x_dim']
        self.y_dim = config['y_dim']
        self.num_funcs = config['meta_batch_size']
        # Normalizing constant
        self.pos_var = config['pos_var']
        self.vel_var = config['vel_var']
        self.q_mean  = config['q_mean']
        self.q_var   = config['q_var']
    
    def read_dataset(self, N):
        """
        Reads the N-th dataset
        """
        filename = self.dir + str(N) + '.yml'
        f = open(filename)
        data = yaml.load(f)
        f.close()
        return data

    def normalize_x(self, x):
        """
        Manually normalize x and y to speed up training.
        This is not ideal, but in practice improves the convergence
        on ALPacA
        """
        x_norm = np.zeros(x.shape)
        x_norm[:,:,0] = x[:,:,0] / self.pos_var 
        x_norm[:,:,1] = x[:,:,1] / self.vel_var
        return x_norm

    def normalize_y(self, y):
        y_norm = (y - self.q_mean) / self.q_var
        return y_norm

    def denormalize_x(self, x_norm):
        x = np.zeros(x_norm.shape)
        x[:,:,0] = x_norm[:,:,0] * self.pos_var
        x[:,:,1] = x_norm[:,:,1] * self.vel_var
        return x

    def denormalize_y(self, y_norm):
        y = y_norm * self.q_var + self.q_mean
        return y

    def sample(self, n_funcs, n_samples):
        assert n_funcs == self.num_funcs
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))
        
        # Load data from file to X and Y
        data_ind = np.random.randint(self.num_data)
        X, Y = self.read_dataset(data_ind)
        T = X.shape[1]
        # Sample num_samples to output
        if n_samples > T:
            raise ValueError('You are requesting more samples \
                             than are in the dataset.')
        inds_to_keep = np.random.choice(T, n_samples)
        np.copyto(x, self.normalize_x(X[:,inds_to_keep,:]))
        np.copyto(y, self.normalize_y(Y[:,inds_to_keep,:]))
        #np.copyto(y, Y[:,inds_to_keep,:])
        
        return x,y
