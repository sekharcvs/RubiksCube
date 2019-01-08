import dqn_network
import cube
import numpy as np
import evaluate_model

## Random Notes:
# We are using TD learning
# The estimate of the updated value function given our policy is a biased estimate (as we bootstrap)
# To avoid this, start with a pretty decent policy (a policy which is more deterministic and algorithmic) - Example Q learning or a DNN
# And because of this policy we know that in most cases, the step taken from the current state will lead to a better state (closer to the solved state)
# Start estimating value function (and policy update) from 1-step away, then go to 2-steps away and so on...
# This way, we know that the TD target (which may be still biased), is less biased as the true value of 1 step away has already converged pretty well when we are estimating the value 2 steps away
# You need scheduled step size which decreases with episodes for better convergence
# TD implicitly assumes markov property (faster learning) and MC doesnt assume markov property (slower learning).
# Under the hood, every process in the world is an MDP if the state is exhaustive enough. It is just our inability to observe the state fully which makes the process non-MDP.
# Use TD for learning in envs closer to Markov and use MC for learning in non-Markov situations
# In real life, when your rewards in future are independent of where you come from, use TD, else use MC.
# Convert situations into MDPs and use TD for faster growth. (look at lecture #4 from David Silver to better understand and write a blog)
# There are also Hybrid (MC+TD) learning methods. And the mix is dependent on how far away from Markov property the env behaves.
# Under the hood, there is always an MDP. Your observations != state. Hence, the MDP assumption breaks down sometimes.
# So, IRL, improve your observation power and use that to learn faster using TD. Every human is an MDP, you just need to observe them better.
# TD-N is the optimal dependent on situation choose N. You propagate info faster with N being > 1, but we compromise the Markov property assumption more and more as N rises.
# Next steps: Read up TD(lambda) algorithm and also eligibility graphs and how Forward view TD(lambda) is equivalent to backward view of eligibility.

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

LOAD_NETWORK = False
SAVE_NETWORK = False
UPDATE_NETWORK = True

LOAD_NETWORK_NAME = "DQN_2STEP_5000_episodes.pickle"
SAVE_NETWORK_NAME = "DQN_2STEP_20000_episodes.pickle"

## Q-network specific parameters ##
EPSILON_START = 0.0 # Explore 50% of the time
EPSILON_END = 0.05 # Greedy policy
N_EPISODES = 15000 # Total Number of episodes to be run
TRAIN_PER_EPISODES = 200 # Training epoch run after collecting this many episodic data


LEARNING_RATE = 0.01 # Q-Network trainer learning_rate
DISCOUNT_FACTOR = 0.99 # Q-Network trainer discount_factor

# Set n_steps_episode_max = -1 for an episode structure which doesnt terminate in fixed moves count but terminates in a state
N_STEPS_EPISODE_MAX = 20 # Max number of steps taken per episode.

###################################################

## Environment to Q-Network interface parameters ##
SIDE = 2
N_MOVES_AWAY_MAX = N_STEPS_EPISODE_MAX
###################################################


## Q Network specific Functions ##

##################################

## Ideally all env specific functions would be in the env object


## Interface between Q Network and Environment ##

def reset_env():
    ## Set Parameters ##
    side = SIDE
    n_moves = N_MOVES_AWAY_MAX
    ####################

    return cube.CubeObject(side, n_moves)



#################################################


## Q Network specific functions ##

def run_episode(q_network_obj, epsilon=1):
    # Set Parameters ##
    n_steps_episode_max = N_STEPS_EPISODE_MAX
    ###################


    env = reset_env()
    obs = env.get_observation()

    steps_taken = 0

    state_action_reward_list = []
    while True:
        action_idx, _ = dqn_network.policy(q_network_obj, np.reshape(obs, (1, -1)), epsilon)
        action_idx = action_idx[0][0]

        terminate_episode = env.is_terminal_state()

        if n_steps_episode_max != -1 and steps_taken >= n_steps_episode_max:
            terminate_episode = True

        if terminate_episode is True:
            break

        obs_new, reward = env.apply_action(action_idx)

        state_action_reward_list += [[obs, obs_new, action_idx, reward]]
        steps_taken = steps_taken + 1

    return state_action_reward_list

def update_q_table(dqn_network_obj, state_action_reward_list, alpha, gamma):
    # Form Input and output lists
    N = len(state_action_reward_list)

    for i in range(len(state_action_reward_list)):
        s0 = state_action_reward_list[i][0]
        s1 = state_action_reward_list[i][1]
        a = state_action_reward_list[i][2]
        r = state_action_reward_list[i][3]

        X_ = s0
        _, Q = dqn_network.policy(dqn_network_obj, np.reshape(s1, (1, -1)), 1)
        Q = Q[0]
        V_s1 = max(Q)
        y_ = Q
        y_[a] = r + gamma * V_s1

        if i == 0:
            X = np.zeros((N, s0.size))
            y = np.zeros((N, y_.size))
        X[i] = X_
        y[i] = y_



    dqn_network.train_op(dqn_network_obj, X, y, alpha)



def run():
    # Main Q-Network former and trainer

    ## Set Parameters ##
    n_inputs = reset_env().get_observation().shape[0]
    n_actions = reset_env().n_actions
    n_episodes = N_EPISODES
    train_per_episodes = TRAIN_PER_EPISODES

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
        dqn_network_obj = dqn_network.dqn_network(n_inputs=n_inputs, n_actions=n_actions)
    else:
        dqn_network_obj = dqn_network.load_network(load_network_name)

    if update_network is True:
        # Run episodes
        state_action_reward_list = []
        for episode in range(n_episodes):
            epsilon = dqn_network.epsilon_control_algo(epsilon_start, epsilon_end, n_episodes, episode)
            gamma = gamma_default
            alpha = alpha_default

            if episode % train_per_episodes == 0:
                # Reset training data every train_per_episodes episodes including 0th episode
                state_action_reward_list = []

            state_action_reward_list += run_episode(dqn_network_obj, epsilon)

            if episode % train_per_episodes == (train_per_episodes - 1):
                # Run training epoch after collecting train_per_episodes episodic data
                update_q_table(dqn_network_obj, state_action_reward_list, alpha, gamma)
                dqn_network.save_network(dqn_network_obj, 'temp_validation_network.pickle')
                evaluate_model.run('temp_validation_network.pickle')

    if save_network is True:
        dqn_network.save_network(dqn_network_obj, save_network_name)

if __name__ == "__main__":
    run()











