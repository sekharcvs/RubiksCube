import dqn_network as network
import numpy as np

DEFAULT_MODEL_NAME = "temp_validation_network.pickle"

def get_action(obs, dqn_obj):
    return network.policy(dqn_obj, np.reshape(obs, (1, -1)), epsilon=1)[0][0], True

def network_step(obs, network_obj = -1):
    # Load Default Network if nothing specified
    if network_obj == -1:
        network_obj = network.load_network(DEFAULT_MODEL_NAME)

    action_idx, does_state_exist = get_action(obs, network_obj)

    return np.asarray([action_idx]), does_state_exist