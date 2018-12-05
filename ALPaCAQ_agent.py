import numpy as np
import tensorflow as tf
from main.alpaca import ALPaCA
from MCDataset import MCOfflineDataset

class ALPaCA_Q():
    """
    The online part (agent) of the ALPaCA Q learning
    Takes an environment and a pretrained Alpaca agent and can be trained
    with either e-greedy or TS algorithm
    """
    def __init__(self, alpaca, env, config, session):
        self.env = env
        self.config = config
        self.data = MCOfflineDataset(config)
        # Multiplier to account for normalized prediction
        self.q_var_mult = config['q_var']
        # Extract info from trained alpaca agent
        self.alpaca = alpaca
        K0 = session.run(alpaca.K).astype(np.float64)
        L0 = session.run(alpaca.L).astype(np.float64)
        L_inv = np.linalg.inv(L0)
        self.Q = L0 @ K0
        self.SigEps = session.run(alpaca.SigEps)[0,0,0,0]
        self.K = K0.copy()
        self.L_inv = np.tile(L_inv, (3,1,1))
        # Training
        self.resample = 80
        self.update_target = 500
        self.num_steps = 0
        self.num_episodes = 0
        self.sampled_K = np.zeros(K0.shape, dtype=np.float64)
        self.target_K = K0.copy()
        self.target_L_inv = self.L_inv.copy()
        self.sample_last_layer()

    def encode_observation(self, observation):
        """
        Map an observation to phi
        """
        norm_ob = self.data.normalize_x(observation[None,None,:])
        phi = self.alpaca.encode(norm_ob)[0].T
        return phi

    def decode_return(self, pred):
        """
        Map a prediction to q-value
        """
        g = self.data.denormalize(pred[None,None,:])
        return g

    def predict_q_values(self, state):
        """
        Given an observation, return the q value predicted by the alpaca
        agent
        observation should be a 1-d array
        """
        phi = self.encode_observation(state)
        mu = self.data.denormalize_y(self.K.T @ phi)
        #var = (1 + phi.T @ self.L_inv[action] @ phi) * self.SigEps * \
            #(dataset.q_var ** 2)
        return mu

    def predict_var(self, state):
        phi = self.encode_observation(state)
        var = np.zeros(self.L_inv.shape[0])
        for i in range(3):
            var[i] = (1 + phi.T @ self.L_inv[i] @ phi) * self.SigEps \
                * (self.q_var_mult ** 2)
        return var


    def predict_sampled_q_values(self, state):
        """
        Return the q-values predicted from current sampled last layer
        """
        phi = self.encode_observation(state)
        mu = self.data.denormalize_y(self.sampled_K.T @ phi)
        return mu

    def predict_target_q_values(self, state):
        """
        Predict q-values based on the target weights
        """
        phi = self.encode_observation(state)
        mu = self.data.denormalize_y(self.target_K.T @ phi)
        return mu

    def update_model(self, state, action, target):
        """
        Update the parameters based on the observed state transition and reward
        """
        L_inv = self.L_inv[action]
        Q = self.Q[:,action]
        phi = self.encode_observation(state)
        # update lambda Alg.2, line 4
        L_phi = L_inv @ phi
        L_inv_new = L_inv - (1 / (1 + phi.T @ L_phi)) * (L_phi @ L_phi.T)
        Q_new = Q[:, None] + phi * target
        self.K[:, action] = (L_inv_new @ Q_new)[:, 0]
        self.L_inv[action] = L_inv_new
        self.Q[:, action] = Q_new[:, 0]

    def sample_last_layer(self):
        """
        Sample a K matrix
        """
        self.sampled_K[:, 0] = np.random.multivariate_normal(
            self.target_K[:, 0], self.target_L_inv[0] * self.SigEps)
        self.sampled_K[:, 1] = np.random.multivariate_normal(
            self.target_K[:, 1], self.target_L_inv[1] * self.SigEps)
        self.sampled_K[:, 2] = np.random.multivariate_normal(
            self.target_K[:, 2], self.target_L_inv[2] * self.SigEps)

    def ts_train_episode(self, step_limit, render=False):
        """
        TS training per episode
        """
        self.sample_last_layer()
        # Initialize env variables
        observation = self.env.reset()
        done = False
        step = 0
        while (step < step_limit) and not done:
            sampled_q_values = self.predict_sampled_q_values(observation)
            action = np.argmax(sampled_q_values)
            if render:
                self.env.render()
            new_ob, reward, done, _ = self.env.step(action)
            if not done:
                next_q_value = self.predict_target_q_values(new_ob).max()
                next_q_value = min(next_q_value, 0)
                target = self.data.normalize_y(reward + next_q_value)
                self.update_model(observation, action, target)
            else:
                target = self.data.normalize_y(0)
                self.update_model(observation, action, target)
            observation = new_ob
            step = step + 1
            self.num_steps += 1
            #if self.num_steps % self.resample == 0:
                #self.sample_last_layer()
            if self.num_steps % self.update_target == 0:
                np.copyto(self.target_K, self.K)
                np.copyto(self.target_L_inv, self.L_inv)
        return step

    def ts_non_bootstrap(self, step_limit):
        # Initialize agent
        self.sample_last_layer()
        # Initialize env variables
        observation = self.env.reset()
        done = False
        step = 0
        # Initialize an array to store the trajectory
        
        while (step < step_limit) and not done:
            sampled_q_values = self.predict_sampled_q_values(observation)
            action = np.argmax(sampled_q_values)
            if render:
                self.env.render()
            new_ob, reward, done, _ = self.env.step(action)
            if not done:
                next_q_value = self.predict_target_q_values(new_ob).max()
                next_q_value = min(next_q_value, 0)
                target = self.data.normalize_y(reward + next_q_value)
                self.update_model(observation, action, target)
            else:
                target = self.data.normalize_y(0)
                self.update_model(observation, action, target)
            observation = new_ob
            step = step + 1
            self.num_steps += 1
            if self.num_steps % self.update_target == 0:
                self.target_K = self.K.copy()
                self.target_L_inv = self.L_inv.copy()
        return step

    def ts_train(self, step_limit=300, num_episode=200):
        for i in range(num_episode):
            step = self.ts_train_episode(step_limit)
            print('Game lasted', step, 'steps')
            self.num_episodes += 1

    def e_greedy_train_episode(self, step_limit, epsilon=0.3, render=False):
        # Initialize env variables
        observation = self.env.reset()
        done = False
        step = 0
        while (step < step_limit) and not done:
            pred_q_values = self.predict_q_values(observation)
            action = np.argmax(pred_q_values)
            if np.random.random() < epsilon:
                action = np.random.randint(3)
            if render:
                self.env.render()
            new_ob, reward, done, _ = self.env.step(action)
            if not done:
                next_q_value = self.predict_target_q_values(new_ob).max()
                next_q_value = min(next_q_value, 0)
                target = self.data.normalize_y(reward + next_q_value)
                self.update_model(observation, action, target)
            else:
                target = self.data.normalize_y(0)
                self.update_model(observation, action, target)
            observation = new_ob
            step = step + 1
            self.num_steps += 1
            if self.num_steps % self.update_target == 0:
                self.target_K = self.K.copy()
                self.target_L_inv = self.L_inv.copy()
        return step

    def e_greedy_train(self, step_limit=300, num_episode=200):
        pass

    def get_action(self, observation, training=False):
        pred_values = self.predict_q_values(observation)
        action = np.argmax(pred_values)
        return action

# Helper functions
def load_alpaca_agent(path):
    """
    Load a pretrained alpaca model stored at path
    """
    g1 = tf.Graph()
    sess1 = tf.InteractiveSession(
        config=tf.ConfigProto(log_device_placement=True), graph=g1)
    agent = ALPaCA(config, sess1, g1)
    agent.construct_graph()
    agent.restore(path)
    return agent
