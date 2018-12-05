import gym
import numpy as np
from gym import wrappers

# Make an environment
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')

def train(env, agent, it=100):
    """
    Rollout the training
    """
    ep_lengths = []
    for i in range(it):
        observation = env.reset()
        done = False
        cnt = 0
        # Rollout
        while not done:
            # Select action from the get_action function
            action = agent.get_action(observation)
            observation, reward, done, _ = env.step(action)
            cnt += 1
        ep_lengths.append(cnt)
    # Episode ends; post-episode updates
    agent.post_episode_update(np.mean(ep_lengths))

def play_episode(env, agent, it=400, render=True):
    observation = env.reset()
    done = False
    cnt = 0
    while (not done) and (cnt <= it):
        if render:
            env.render()
        cnt += 1
        action = agent.get_action(observation,False)
        observation, reward, done, _ = env.step(action)
    return cnt

def get_VI_trajectory(env, VIagent, it=400):
    observation = env.reset()
    done = False
    cnt = 0
    s_traj = []
    a_traj = []
    q_traj = []
    while (not done) and (cnt <= it):
        cnt += 1
        value = VIagent.get_value(observation) 
        action = VIagent.get_action(observation, False)
        s_traj.append(observation)
        a_traj.append(action)
        q_traj.append(value[action])
        observation, reward, done, _ = env.step(action)
    print('game lasted', cnt, 'moves')
    s_traj = np.array(s_traj).T
    a_traj = np.array(a_traj).T
    q_traj = np.array(q_traj).T
    return np.vstack([s_traj, a_traj, q_traj]).T
