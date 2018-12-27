import numpy as np
import sys
import random
import pickle

# Big assumption: States and observations are one to one.

def default_existence_method(list, elem):
    return elem in list

def default_index_method(list, elem):
    return list.index(elem)

def get_existance(list, elem, method=default_existence_method):
    # Checking existence in a list is not straightforward with numpy arrays. Hence we need a function pointer from the environment
    return method(list, elem)

def get_index(list, elem, method_index=default_index_method, method_existence=default_existence_method):
    if method_existence(list, elem) is False:
        return -1
    else:
        return method_index(list, elem)

def argmax(a):
    # Finds the argmax of the list a.
    # If more than one maxima exist, pick one index randomly
    idx = np.argmax(a)
    v = a[idx]
    idx = [index for index, value in enumerate(a) if value == v]
    return random.choice(idx)

def epsilon_control_algo(eps_start, eps_end, episode_max, episode):
    eps = eps_start + (eps_end - eps_start) * (episode)/episode_max
    lower = min(eps_start, eps_end)
    upper = max(eps_start, eps_end)
    return max(min(eps, upper), lower)

def policy(q_table, state_idx, epsilon=1):
    # Epsilon greedy policy implementation
    n_actions = q_table.shape[1]
    r = random.random()

    greedy_idx = argmax(q_table[state_idx])
    if epsilon >= r:
        # Do the greedy action
        action_idx = greedy_idx

    else:
        # Randomly select among the rest of the actions
        action_idx = random.randrange(0, n_actions - 1)
        if action_idx > greedy_idx:
            action_idx = action_idx + 1

    return action_idx

def add_state_q_table(q_table):
    n_states = q_table.shape[0]
    n_actions = q_table.shape[1]
    #q_table_new = np.zeros([n_states+1, n_actions])
    #q_table_new[0:n_states][0:n_actions] = q_table[:][:]

    ## Assign equal state-action values for the new state
    #q_table_new[n_states][:] = 0

    q_table_new = np.vstack([q_table, np.zeros([1, n_actions])])

    return q_table_new

def add_state_state_obs_list(state_obs_list, obs):
    state_obs_list.append(obs)
    return state_obs_list

def add_state(q_table, state_obs_list, obs):
    q_table = add_state_q_table(q_table)
    state_obs_list = add_state_state_obs_list(state_obs_list, obs)

    return q_table, state_obs_list

def get_best_action_state(q_table, state):
    return(np.argmax(q_table[state]))

def get_best_action_observation(q_table, state_obs_list, obs):
    state = state_obs_list.index(obs)
    return get_best_action_state(q_table, state)

def get_q_network_obj(model_name):
    return pickle.load(open(model_name, 'rb'))

def dump_q_network(q_network_object, model_name):
    pickle.dump(q_network_object, open(model_name, 'wb'))

class q_network:
    def __init__(self, n_states, n_actions):
        # Have a state to observation mapping table
        # Have a state, action and value table
        # have a number of states parameters
        # have a number of possible actions
        self.n_states = n_states
        self.n_actions = n_actions
        self.qtable = np.zeros([self.n_states, self.n_actions])
        self.state_obs_list = [0]*n_states

    def add_state(self, obs):
        self.qtable, self.state_obs_list = add_state(self.qtable, self.state_obs_list, obs)
        self.n_states = self.n_states + 1

    def get_best_action_observation(self, obs):
        return get_best_action_observation(self.qtable, self.state_obs_list, obs)

    def get_best_action_state(self, state):
        return get_best_action_state(self.qtable, state)

    def get_n_states(self):
        return self.qtable.shape[0]

    def get_n_actions(self):
        return self.qtable.shape[1]

    def fetch_q_table(self, qtable):
        return self.qtable

    def update_q_table(self, qtable):
        if qtable.shape[0] != self.n_states or qtable.shape[1] != self.n_actions:
            print("Update q table: new q table\'s dimensions don\'t match existing table")
            exit(-1)
        else:
            self.qtable = qtable