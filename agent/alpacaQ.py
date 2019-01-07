##################################################################
# Fengjun Yang, 2018
# Code for doing q-learning with the alpaca algorithm
##################################################################

import numpy as np
import tensorflow as tf
from metamountaincar.mmcNormalize import mmcNorm

class ALPaCAQ():
    """
    The online part (agent) of the ALPaCA Q learning
    Takes an environment and a pretrained Alpaca agent and can be trained
    with either e-greedy or TS algorithm
    """
    def __init__(self, offline, env):
        self.env = env
        self.config = offline.config
        self.norm = mmcNorm(offline.config)
        self.alpaca = offline.alpaca
        # Extract the parameters
        session = offline.sess
        alpaca = offline.alpaca
        K0 = session.run(alpaca.K).astype(np.float64)
        L0 = session.run(alpaca.L).astype(np.float64)
        L_inv = np.linalg.inv(L0)
        self.Q = L0 @ K0
        self.SigEps = session.run(alpaca.SigEps)[0,0,0,0]
        self.K = K0.copy()
        self.L_inv = np.tile(L_inv, (3,1,1))
        # Store the initial weights so we can reset the agent in experiments
        self.K0 = K0
        self.L_inv_0 = self.L_inv
        self.Q0 = self.Q

    def reset(self):
        self.K = self.K0
        self.L_inv = self.L_inv_0
        self.Q = self.Q0

    def predict_q_values(self, state, K=None):
        """
        Given an observation, return the q value predicted by the alpaca
        agent. Observation should be a 1-d array
        Can specify K to be the sampled weights or target weights
        """
        if K is None:
            K = self.K
        phi = self.encode_observation(state)
        mu = self.norm.denorm_y(K.T @ phi)
        return mu

    def predict_var(self, state):
        """
        Give the variance of prediction
        """
        phi = self.encode_observation(state)
        var = np.zeros(self.L_inv.shape[0])
        for i in range(3):
            v = (1 + phi.T @ self.L_inv[i] @ phi) * self.SigEps
            var[i] = self.norm.denorm_var_pred(v)
        return var

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
        Q_new = Q[:, None] + phi * self.norm.norm_y(target)
        self.K[:, action] = (L_inv_new @ Q_new)[:, 0]
        self.L_inv[action] = L_inv_new
        self.Q[:, action] = Q_new[:, 0]

    def get_action(self, observation, training=False):
        pred_values = self.predict_q_values(observation)
        action = np.argmax(pred_values)
        return action

    def encode_observation(self, observation):
        """
        Map an observation to phi
        """
        norm_ob = self.norm.norm_x(observation[None,None,:])
        phi = self.alpaca.encode(norm_ob)[0].T
        return phi

    def decode_return(self, pred):
        """
        Map a prediction to q-value
        """
        q = self.norm.denorm_y(pred[None,None,:])
        return q

class EGreedyALPaCAQ(ALPaCAQ):
    """
    ALPaCAQ agent that explores the environment with epsilon-greedy behavior
    policy
    """
    def __init__(self, offline, env):
        super().__init__(offline, env)
        self.K_target = self.K.copy()
        self.L_inv_target = self.L_inv.copy()
        self.update_target = 500
        self.num_steps = 0

    def train(self, epsilon=0.3, step_limit=200, render=False):
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
            # Update model
            if done:
                target = 0
            else:
                next_q_value = self.predict_q_values(
                    new_ob, self.K_target).max()
                next_q_value = min(next_q_value, 0)
                target = reward + next_q_value
            self.update_model(observation, action, target)
            # Step
            observation = new_ob
            step = step + 1
            self.num_steps += 1
            if self.num_steps % self.update_target == 0:
                self.K_target = self.K.copy()
                self.L_inv_target = self.L_inv.copy()
        return step

class TS_ALPaCAQ(ALPaCAQ):
    def __init__(self, offline, env):
        super().__init__(offline, env)
        self.resample = 80
        self.update_target = 500
        self.num_steps = 0
        self.num_episodes = 0
        self.K_sample = np.zeros(self.K.shape, dtype=np.float64)
        self.K_target = self.K.copy()
        self.L_inv_target= self.L_inv.copy()
        self.sample_last_layer()

    def sample_last_layer(self):
        """ Sample a last layer weights """
        self.K_sample[:, 0] = np.random.multivariate_normal(
            self.K_target[:, 0], self.L_inv_target[0] * self.SigEps)
        self.K_sample[:, 1] = np.random.multivariate_normal(
            self.K_target[:, 1], self.L_inv_target[1] * self.SigEps)
        self.K_sample[:, 2] = np.random.multivariate_normal(
            self.K_target[:, 2], self.L_inv_target[2] * self.SigEps)

    def train(self, step_limit=200, render=False):
        """
        TS training per episode; Resample weights every episode
        """
        self.sample_last_layer()
        # Initialize env variables
        observation = self.env.reset()
        done = False
        step = 0
        while (step < step_limit) and not done:
            sampled_q_values = self.predict_q_values(
                observation, self.K_sample)
            action = np.argmax(sampled_q_values)
            if render:
                self.env.render()
            new_ob, reward, done, _ = self.env.step(action)
            if done:
                target = 0
            else:
                next_q_value = self.predict_q_values(
                    new_ob, self.K_target).max()
                next_q_value = min(next_q_value, 0)
                target = reward + next_q_value
            self.update_model(observation, action, target)
            observation = new_ob
            step = step + 1
            self.num_steps += 1
            if self.num_steps % self.update_target == 0:
                np.copyto(self.K_target, self.K)
                np.copyto(self.L_inv_target, self.L_inv)
        return step

class TS_MC_ALPaCAQ(TS_ALPaCAQ):
    def __init__(self, offline, env):
        super().__init__(offline, env)

    def ts_sample_traj(self, step_limit):
        """
        Samples a trajectory to use for Monte-Carlo training
        """
        sa_traj, reward_traj = [], []
        observation = self.env.reset()
        counter = 0
        done = False
        # Sample a trajectory
        while counter < step_limit and not done:
            # sample action based on sampled last layer
            sampled_q_values = self.predict_sampled_q_values(observation)
            action = np.argmax(sampled_q_values)
            sa_traj.append((observation, action))
            observation, reward, done, _ = self.env.step(action)
            reward_traj.append(reward)
            counter = counter + 1
        # convert reward to target values
        target_val = np.cumsum(reward_traj[::-1])[::-1]
        return sa_traj, target_val

    def train(self, step_limit=200):
        # Resample last layer every episode
        self.sample_last_layer()
        # Sample a trajectory and find target values in the traj
        sa_traj, target_val = self.ts_sample_traj(step_limit)
        episode_length = len(target_val)
        for i in range(episode_length):
            self.update_model(*sa_traj[i], target_val[i])
        return episode_length
