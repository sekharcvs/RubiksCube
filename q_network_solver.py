import q_network
import cube
import random
import numpy as np

## Load Model Name ##
DEFAULT_MODEL_NAME = "Q_Network_2000_episodes.pickle"
#####################

def get_action(obs, qnetwork_obj):
    qnetwork = qnetwork_obj.qtable
    state_obs_list = qnetwork_obj.state_obs_list

    state_idx = q_network.get_index(state_obs_list, obs, cube.method_index_numpy_arrays, cube.method_existence_numpy_arrays)
    if state_idx != -1:
        return q_network.policy(qnetwork, state_idx, epsilon=1), True
    else:
        n_actions = qnetwork_obj.n_actions
        return random.randrange(0, n_actions - 1), False

def network_step(obs, qnetwork_obj = -1):
    # Load Default Network if nothing specified
    if qnetwork_obj == -1:
        qnetwork_obj = q_network.load_network(DEFAULT_MODEL_NAME)

    action_idx, does_state_exist = get_action(obs, qnetwork_obj)

    return np.asarray([action_idx]), does_state_exist