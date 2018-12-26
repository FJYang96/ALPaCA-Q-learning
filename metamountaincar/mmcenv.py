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

    def __init__(self, thrust=0.001, gravity=0.0025):
        super().__init__()
        self.thrust = thrust
        self.gravity = gravity


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

