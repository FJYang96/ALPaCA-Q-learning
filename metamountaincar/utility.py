############################################
# Fengjun Yang, 2018
# This file contains some experiment and plotting utility functions
# for the mountain car environment.
############################################

from metamountaincar.VI import mmcVI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm

def play_episode(env, agent, it=400, render=True):
    """
    Let an agent act in the environment for one episode and get back the
    step count
    Input:
        - env: the environment (compatible with the gym.env class)
        - agent: an agent that has a get_action(observation) method
        - it: the maximum number of steps per episode
        - render: whether or not to render the episode
    Return:
        - length of the episode
    """
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

def learning_rate(agent, env, num_experiments=10, num_episodes=1000,
                  num_test_plays=3, step_limit=200):
    """
    Input:
        - agent:            ALPaCAQ agent
        - env:              the experiment environment
        - num_experiment:   repeat the same experiment how many times
        - num_episode:      how many episodes do we train the agent for
        - num_test_plays:   how many times do we deploy the agent in the real
                            environment
        - step_limit:       max number of steps per episode
    """
    result = np.zeros((num_experiments, num_episodes))
    for i in tqdm(range(num_experiments)):
        agent.reset()
        for j in range(num_episodes):
            train_step = agent.train(step_limit=step_limit,render=False)
            # Play several games to find average steps
            play_step = []
            for k in range(num_test_plays):
                play_step.append(play_episode(env, agent, render=False,it=step_limit))
            result[i][j] = np.mean(play_step)
    return result

def plot_learning_rate(result):
    """
    Takes in the learning rate return and make a plot of the learning rate
    """
    avg_result = result.mean(0)
    ind = np.arange(len(avg_result)) + 1
    plt.plot(ind, avg_result)
    plt.show()

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

def ctg_table_helper(table, savedir=None):
    """
    Helper function for plotting the cost to go function in the meta mountain
    car environment, given the ctg table.
    """
    # Process the data for plotting
    x_gran, y_gran = table.shape[0], table.shape[1]
    X = np.arange(-1.2, 0.6, 1.8/x_gran)
    Y = np.arange(-0.07, 0.07, 0.14/y_gran)
    X,Y = np.meshgrid(X,Y)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,table, cmap=cm.coolwarm)
    if savedir is not None:
        fig.savefig(savedir,transparent=True)

def ctg_func_helper(func, gran, savedir=None):
    """
    Helper function for plotting the cost-to-go function in the meta mountain
    car environment. Take as input a function that 
    Input:
        - func: the function to plot
        - gran: the fineness of the grid
    """
    # Fill in the table by iteratively querying the agent
    X = np.arange(-1.2, 0.6, 1.8/gran)
    Y = np.arange(-0.07, 0.07, 0.14/gran)
    X,Y = np.meshgrid(X,Y)
    Z = np.zeros(X.shape)
    for i in range(gran):
        for j in range(gran):
            Z[i,j] = -func(np.array([X[i,j],Y[i,j]]))
    ctg_table_helper(Z, savedir)

def plot_true_cost_to_go(env, gran, it, savedir=None):
    """
    Plot the cost-to-go found by value iteration
    Input:
        - env: a meta mountaincar environment
        - gran: granularity of the VI algorithm
        - it: max number of iterations for the VI algorithm
    """
    VIagent = mmcVI(gran, it, gravity=env.gravity, thrust=env.thrust)
    VIagent.value_iteration()
    Q_table = VIagent.q_table
    V_table = np.max(Q_table, axis=2)
    ctg_table_helper(-V_table, savedir)

def plot_learned_ctg(agent, gran, savedir=None):
    """
    Plot the state value function
    """
    func = lambda x: agent.predict_q_values(x).max()
    ctg_func_helper(func, gran, savedir)

def plot_variance(agent, gran, savedir=None):
    """
    Plot the mean variance for all the actions of a state
    """
    func = lambda x: agent.predict_var(x).mean()
    ctg_func_helper(func, gran, savedir)

