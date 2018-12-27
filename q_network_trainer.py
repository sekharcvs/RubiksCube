import q_network
import cube

# This trainer should be state, action, env agnostic
# Hyper parameters can be specified, but environment should be a passable parameter

# For DQN/DNN/RNNs, think of embedding the states using something like word2vec - this might help group states.
# Examples - (You could even do multitask)
# train an embedding to predict how far away from solved state.
# train an embedding to predict the middle state between two states
# train an embedding to predict all the rotated forms of a cube state

## Steps for my understanding
# Start with an epsilon greedy approach (hyperparameter), slowly prune it down
# Have a stochastic policy function PI which takes in a Q-table and an epsilon
# Q(s, a) = Q(s, a) + learning_rate * (r(s, a) + discount * value(s')_PI - Q(s, a))

## HYPER-PARAMETERS ##

PRINT_DIAG = True

LOAD_NETWORK = True
SAVE_NETWORK = True
UPDATE_NETWORK = True

LOAD_NETWORK_NAME = "Q_Network_2STEP_5000_episodes.pickle"
SAVE_NETWORK_NAME = "Q_Network_2STEP_20000_episodes.pickle"

## Q-network specific parameters ##
EPSILON_START = 0.0 # Explore 50% of the time
EPSILON_END = 0.05 # Greedy policy
N_EPISODES = 15000

LEARNING_RATE = 0.01 # Q-Network trainer learning_rate
DISCOUNT_FACTOR = 1 # Q-Network trainer discount_factor
###################################

## Cube specific parameters ##
SIDE = 2
N_MOVES = 2

##############################

## Environment to Q-Network interface parameters ##
N_ACTIONS = 3*SIDE*2

# Set n_steps_episode_max = -1 for an episode structure which doesnt terminate in fixed moves count but terminates in a state
N_STEPS_EPISODE_MAX = N_MOVES # Max number of steps taken per episode.

###################################################


## Q Network specific Functions ##

##################################

## Ideally all env specific functions would be in the env object


## Interface between Q Network and Environment ##

def reset_env():
    ## Set Parameters ##
    side = SIDE
    n_moves = N_MOVES
    ####################

    return cube.CubeObject(side, n_moves)



#################################################


## Q Network specific functions ##

def run_episode(q_network_obj, episode, epsilon=1):
    # Set Parameters ##
    n_episodes = N_EPISODES
    n_steps_episode_max = N_STEPS_EPISODE_MAX
    ###################

    if PRINT_DIAG is True:
        n_states_start = q_network_obj.n_states
        n_states_new = 0

    env = reset_env()
    obs = env.get_observation()
    #if (obs in q_network_obj.state_obs_list) is False:
    if q_network.get_existance(q_network_obj.state_obs_list, obs, method=env.method_existence) is False:
        # New observation - add it
        q_network_obj.add_state(obs)
        if PRINT_DIAG is True:
            n_states_new = n_states_new + 1

    terminate_episode = False
    steps_taken = 0

    state_action_reward_list = []
    while True:
        state_obs_list = q_network_obj.state_obs_list
        #state_idx = state_obs_list.index(obs)
        state_idx = q_network.get_index(state_obs_list, obs, method_existence=env.method_existence,
                                            method_index=env.method_index)
        qtable = q_network_obj.qtable
        action_idx = q_network.policy(qtable, state_idx, epsilon)

        terminate_episode = env.is_terminal_state()

        if n_steps_episode_max != -1 and steps_taken >= n_steps_episode_max:
            terminate_episode = True

        if terminate_episode is True:
            break

        obs, reward = env.apply_action(action_idx)
        #if (obs in state_obs_list) is False:
        if q_network.get_existance(state_obs_list, obs, method=env.method_existence) is False:
            # New observation - add it
            q_network_obj.add_state(obs)
            state_obs_list = q_network_obj.state_obs_list
            if PRINT_DIAG is True:
                n_states_new = n_states_new + 1

        #state_idx_new = state_obs_list.index(obs)
        state_idx_new = q_network.get_index(state_obs_list, obs, method_existence=env.method_existence, method_index=env.method_index)

        state_action_reward_list += [[state_idx, state_idx_new, action_idx, reward]]
        steps_taken = steps_taken + 1

    if PRINT_DIAG is True:
        print("episode number: {}, number of existing states: {}, number of newly added states: {}".format(episode, n_states_start, n_states_new))

    return q_network_obj, state_action_reward_list

def update_q_table(q_network_obj, state_action_reward_list, alpha, gamma):
    q_table_old = q_network_obj.qtable
    q_table_new = q_table_old.copy()

    if PRINT_DIAG is True:
        n_updates = 0
        total_update_abs = 0

    for i in range(len(state_action_reward_list)):
        item = state_action_reward_list[i]
        s = item[0]
        s_ = item[1]
        a = item[2]
        r = item[3]

        q_table_new[s][a] = q_table_old[s][a] + alpha * (r + gamma * max(q_table_old[s_]) - q_table_old[s][a])

        if PRINT_DIAG is True:
            if q_table_old[s][a] != 0:
                n_updates = n_updates + 1
                total_update_abs = total_update_abs + abs((q_table_new[s][a] - q_table_old[s][a])/q_table_old[s][a])

    if n_updates > 0:
        print("Training stats: number of existing state updates: {}, mean percentage change of weights: {}".format(
            n_updates, 100.0 * total_update_abs / n_updates))
    else:
        # No Updates this episodes
        print("Training stats: number of existing state updates: {}, mean percentage change of weights: {}".format(
            n_updates, "N/A"))

    q_network_obj.update_q_table(q_table_new)


def run():
    # Main Q-Network former and trainer

    ## Set Parameters ##
    n_actions = N_ACTIONS
    n_episodes = N_EPISODES

    epsilon_start = EPSILON_START
    epsilon_end = EPSILON_END

    alpha_default = LEARNING_RATE
    gamma_default = DISCOUNT_FACTOR

    load_network = LOAD_NETWORK
    save_network = SAVE_NETWORK
    update_network = UPDATE_NETWORK

    load_network_name = LOAD_NETWORK_NAME
    save_network_name = SAVE_NETWORK_NAME
    ####################

    # Create a q_network with zero states observed
    if load_network is False:
        q_network_obj = q_network.q_network(n_states=0, n_actions=n_actions)
    else:
        q_network_obj = q_network.get_q_network_obj(load_network_name)

    if update_network is True:
        # Run episodes
        for episode in range(n_episodes):
            epsilon = q_network.epsilon_control_algo(epsilon_start, epsilon_end, n_episodes, episode)
            gamma = gamma_default
            if episode == 0:
                alpha = 1
            else:
                alpha = alpha_default

            q_network_obj, state_action_reward_list = run_episode(q_network_obj, episode, epsilon)
            update_q_table(q_network_obj, state_action_reward_list, alpha, gamma)

    if save_network is True:
        q_network.dump_q_network(q_network_obj, save_network_name)





if __name__ == "__main__":
    run()











