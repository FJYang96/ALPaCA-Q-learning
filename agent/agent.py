########################################################################
# Fengjun Yang, 2018
# Abstract class of agents: defines the methods needed
########################################################################
class Agent:

    def __init__(self, env):
        """
        The environment is currently an attribute of the agent so that
        training can be more easily implemented
        """
        self.env = env
        raise NotImplementedError

    def train(self, num_episode=None):
        """
        Trains the agent for a given number of episodes, or until convergence
        """
        raise NotImplementedError

    def train_episode(self):
        """
        Trains the agent for one episode. Subroutine of train.
        """
        raise NotImplementedError
    
    def get_optimal_action(self, observation):
        """
        Output an action for a given observation
        """
        raise NotImplementedError

    def get_behavior_action(self, observation):
        """
        Output the action given by the behavior policy during training
        """
        raise NotImplementedError
