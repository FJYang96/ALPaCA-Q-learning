from MCTransfer import MountainCarTransfer as MCF
import numpy as np

class MCVI():
    def __init__(self, granularity, step_limit, \
                 thrust=0.001, gravity=0.0025):
        """
        Initialize the value iteration wrapper; build transition table
        """
        # Set environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.A = 3
        self.thrust = thrust
        self.gravity = gravity
        # VI parameters
        self.gran = granularity
        self.step_limit = step_limit
        # The size of each position and velocity bin
        self.pos_size = (self.max_position - self.min_position) / self.gran
        self.vel_size = self.max_speed * 2 / self.gran
        # Initialize the transition + q-value + reward table
        self.t_table = None
        self.q_table = None
        self.r_table = None
        self.build_transition_table()
        self.initialize_r_table()
        self.q_table = np.zeros((self.gran, self.gran, self.A))

    def s_to_ind(self, states):
        """
        Given an array of states and actions, find the q-table index
        """
        pos_ind = (states[:,0] - self.min_position) // self.pos_size
        pos_ind = np.rint(pos_ind).astype(int)
        vel_ind = (states[:,1] + self.max_speed)    // self.vel_size
        vel_ind = np.rint(vel_ind).astype(int)
        #act_ind = np.ones(pos_ind.shape) * action
        return np.stack([pos_ind, vel_ind], axis=1)

    def ind_to_s(self, inds):
        """
        Given an index in the q-/t-table, return corresponding state and
        action
        """
        pos = self.min_position + (inds[:, 0] + 0.5) * self.pos_size
        vel = -self.max_speed + (inds[:, 1] + 0.5) * self.vel_size
        return np.stack([pos, vel], axis=1)

    def step_many_states(self, states, action):
        """
        Given a list of states, apply the transition with action
        """
        orig_pos = states[:, 0].copy()
        orig_vel = states[:, 1].copy()
        pos = states[:, 0]
        vel = states[:, 1]
        vel = orig_vel + (action-1) * self.thrust + \
              np.cos(3*pos) * (-self.gravity)
        vel = np.clip(vel, -self.max_speed, self.max_speed)
        pos += vel
        pos = np.clip(pos, self.min_position, self.max_position)
        # When the car already at left end, set negative speed to zero
        cap = (pos == self.min_position) * (vel < 0)
        vel[cap] = 0
        # If the car already at goal location, undo the changes
        goal = orig_pos >= self.goal_position
        #print(goal, orig_pos[goal])
        pos[goal] = orig_pos[goal]
        vel[goal] = orig_vel[goal]
        return np.stack([pos, vel], axis=1)

    def build_transition_table(self):
        """
        Build the transition table of all (state-action) pairs
        """
        self.t_table = np.zeros((self.gran, self.gran, self.A), dtype=tuple)
        total_bins = self.gran ** 2
        state_inds = np.zeros((total_bins, 2), dtype=np.int64)
        state_inds[:, 0] = (np.arange(total_bins) // self.gran)
        state_inds[:, 0] = np.round(state_inds[:, 0]).astype(int)
        state_inds[:, 1] = np.arange(total_bins) % self.gran
        for action in range(3):
            states = self.ind_to_s(state_inds)
            new_states = self.step_many_states(states, action)
            new_inds = self.s_to_ind(new_states)
            action_inds = np.ones(total_bins, dtype=np.int64) * action
            t_table_inds = np.vstack([state_inds.T, action_inds.T]).T
            for i in range(total_bins):
                self.t_table[tuple(t_table_inds[i])] = tuple(new_inds[i])

    def initialize_r_table(self):
        """
        Initialize the q-table so that the goal states have values 0
        and others 1
        """
        self.r_table = np.ones((self.gran, self.gran, self.A)) * -1
        for ind in range(self.gran):
            pos = self.min_position + (ind + 0.5) * self.pos_size
            if pos >= self.goal_position:
                self.r_table[ind] = 0

    def value_iteration(self):
        """
        Perform value iteration over the q-values
        """
        # TODO: vectorize the code here
        cnt = 0
        while True:
            print("Iteration: ", cnt)
            # For each state
            # Q[pos,vel,a] = max_a (R[pos,vel,a] + Q[pos', vel'])
            new_q_table = np.zeros((self.gran, self.gran, self.A))
            for i in range(self.gran):
                for j in range(self.gran):
                    for action in range(3):
                        new_q_table[i,j,action] = self.r_table[i,j,action] +\
                            np.max(self.q_table[self.t_table[i,j,action]])
            if cnt >= self.step_limit or np.allclose(new_q_table, self.q_table):
                break
            np.copyto(self.q_table, new_q_table)
            cnt = cnt + 1

    def soft_max_decision(self, values):
        """
        Use softmax weigh the q-values and sample an action
        """
        e = np.exp(values - np.max(values))
        prob = e / np.sum(e)
        return np.random.choice(len(prob), p=prob)

    def greedy_decision(self, values):
        """
        Use greedily select actions; randomly break ties
        """
        max_indices = np.flatnonzero( values == values.max() )
        return np.random.choice(max_indices)

    def get_action(self, observation, training=True):
        """
        Get an action to interact with the environment with softmax policy
        """
        ob_q_values = self.get_value(observation)
        action = self.greedy_decision(ob_q_values)
        return action

    def get_value(self, observation):
        """
        Get the value of an observation-action pair
        """
        ob_ind = self.s_to_ind(np.array([observation]))[0]
        ob_ind = tuple(ob_ind)
        return self.q_table[ob_ind]
        
