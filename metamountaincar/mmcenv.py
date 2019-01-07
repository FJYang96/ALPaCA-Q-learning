#####################################################################
# Fengjun Yang, 2018
# This file contains the meta-mountain-car environment. This environment
# is an extension to the mountain car environment where the gravity
# and thrust of the car can be varied.
#####################################################################

from gym.envs.classic_control import MountainCarEnv
import numpy as np

class MetaMountainCar(MountainCarEnv):
    """
    Wrapper class for the Meta mountain car environment
    """

    def __init__(self, config, thrust=0.001, gravity=0.0025):
        super().__init__()
        self.thrust = self._sample_mmc_parameters(config['thrust_range'])
        self.gravity = self._sample_mmc_parameters(config['gravity_range'])

    def _sample_mmc_parameters(self, param_range):
        """
        Sample a parameter based from the range given by param_range
        Input: - param_range: a tuple in the form (lower_bound, upper_bound)
        """
        diff = param_range[1] - param_range[0]
        sample = np.random.random() * diff + param_range[0]
        return sample

    def set_env_param(self, thrust, gravity):
        self.thrust = thrust
        self.gravity = gravity

    def get_env_param(self):
        return (self.thrust, self.gravity)

    def step(self, action):
        """
        Overriding the step function of original MountainCar enviroment
        to incorporate adjusted gravity
        Code adapted from the original openAI gym MountainCar code
        """
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*self.thrust \
                    + np.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

