############################################
# Fengjun Yang, 2018
# This file contains some experiment and plotting utility functions
# for the mountain car environment.
############################################

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

def plot_learned_cost_to_go(ALPaCA_Q, grid, savedir=None):
    """
    Plot the learned cost to go function for an ALPaCA_Q agent. This is designed to
    work with the mountain car environment and does not necessarily generalize to
    other domains of interest
    Input:
        - ALPaCA_Q: a trained alpaca_q agent
        - grid:     finess of the 3D mesh for plotting. Higher values means finer.
        - savedir:  used for saving the figure. If savedir is None, then the plot
                    will not be saved
    """
    # Process the data for plotting
    X = np.arange(-1.2, 0.6, 1.8/grid)
    Y = np.arange(-0.07, 0.07, 0.14/grid)
    X,Y = np.meshgrid(X,Y)
    # Fill in the table by iteratively querying the agent
    Z = np.zeros(X.shape)
    for i in range(grid):
        for j in range(grid):
            Z[i,j] = ALPaCA_Q.predict_q_values(np.array([X[i,j],Y[i,j]])).max()
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,-Z, cmap=cm.coolwarm)
    if savedir is not None:
        fig.savefig(savedir,transparent=True)

def plot_learned_variance(ALPaCA_Q, grid, savedir=None):
    """
    Similar to plot_learned_cost_to_go, except for plotting variance instead
    of mean
    """
    # Process the data for plotting
    X = np.arange(-1.2, 0.6, 1.8/grid)
    Y = np.arange(-0.07, 0.07, 0.14/grid)
    X,Y = np.meshgrid(X,Y)
    # Fill in the table by iteratively querying the agent
    Z = np.zeros(X.shape)
    for i in range(grid):
        for j in range(grid):
            s = np.array([X[i,j],Y[i,j]])
            Z[i,j] = ALPaCA_Q.predict_var(s).mean()
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,np.sqrt(Z), cmap=cm.coolwarm)
    if savedir is not None:
        fig.savefig(savedir,transparent=True)

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
