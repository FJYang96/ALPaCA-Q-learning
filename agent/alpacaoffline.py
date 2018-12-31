###################################################################
# Fengjun Yang, 2018
# Class that wraps offline alpaca q learning. Is parent class for
# agents with different exploration strategies
###################################################################

from agent.alpaca import ALPaCA
import tensorflow as tf

class ALPaCAOffline:
    """
    This class is a wrapper for the alpaca offline training. It will be
    extended to subclasses with different exploration strategies.
    """

    def __init__(self, config):
        """
        Initialize the tensorflow graph and the alpaca agent
        Input:
            - config:   a dictionary of the configuration parameters
        """
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True), graph=self.graph)
        self.alpaca = ALPaCA(config, self.sess, self.graph)
        self.alpaca.construct_model()
        self.config = config

    def train_offline(self, dataset, num_epochs):
        """
        Offline training (supervised learning of q values)
        """
        self.alpaca.train(dataset,num_epochs)

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

    def save_agent(self, savedir):
        self.alpaca.save(savedir)

    def load_agent(self, loaddir):
        # Either load pre-trained agent
        self.alpaca.restore(loaddir)

