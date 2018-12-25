import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.A_space = env.action_space
        self.S_space = env.observation_space

    def online_update():
        pass 

    def get_action(self, observation=None, training=True):
        """
        Return a random action
        """
        action = self.A_space.sample()
        return action


