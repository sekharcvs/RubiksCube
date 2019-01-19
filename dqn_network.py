import numpy as np
import sys
import random
import pickle

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras import optimizers

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

def argmax_multiple(a):
    l = a.shape[0]
    idx = np.zeros(l)
    for i in range(l):
        idx[i] = argmax(a[i])
    return idx

def argmax(a):
    # Finds the argmax of the list a.
    # If more than one maxima exist, pick one index randomly
    idx = np.argmax(a)
    v = a[idx]
    idx = [index for index, value in enumerate(a) if value == v]
    return random.choice(idx)

def train_op(dqn, X, y, alpha, n_epochs):
    model = dqn.model
    sgd = optimizers.SGD(lr=alpha)
    adam = optimizers.Adam(lr=alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.fit(X, y, epochs=n_epochs, verbose=0, shuffle=True)

def policy(dqn, obs, epsilon=1):
    # Epsilon greedy policy implementation
    model = dqn.model
    r = random.random()

    action_values = model.predict(obs)
    n_actions = action_values.shape[-1]

    greedy_idx = argmax_multiple(action_values)
    l = greedy_idx.shape[0]
    if epsilon >= r:
        # Do the greedy action
        action_idx = greedy_idx
    else:
        # Randomly select among the rest of the actions
        action_idx = greedy_idx
        for i in range(l):
            action_idx[i] = random.randrange(0, n_actions - 1)
            if action_idx[i] > greedy_idx[i]:
                action_idx[i] = action_idx[i] + 1

    # Return Action Index and Q Values
    return action_idx.reshape((-1,1)).astype(np.int), action_values

def get_best_action_state(q_table, state):
    return(np.argmax(q_table[state]))

def get_best_action_observation(q_table, state_obs_list, obs):
    state = state_obs_list.index(obs)
    return get_best_action_state(q_table, state)

def load_network(model_name):
    n_inputs, n_actions = pickle.load(open(model_name, 'rb'))
    obj = dqn_network(n_inputs, n_actions)
    obj.model = load_model(model_name+'.h5')
    return obj

def save_network(network_object, model_name):
    model = network_object.model
    a = [network_object.n_inputs, network_object.n_actions]
    pickle.dump(a, open(model_name, 'wb'))
    model.save(model_name+'.h5')

def evaluate(model, test_X, nc):
    return np_utils.to_categorical(model.predict_classes(test_X), nc)

def get_NN(input_shape, n_actions):
    ## Set Parameters ##
    dense_layer_size = 256
    dropout = 0.2
    ####################

    model = Sequential()

    # Fully connected layers
    model.add(Dense(dense_layer_size, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size,  activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_actions, activation='softmax'))

    return model

def accuracy(true, pred):
    a = np.argmax(true, axis = -1)
    b = np.argmax(pred, axis = -1)
    x = np.equal(a, b)
    return np.mean(x)

# def train_epoch(model, train_X, train_y, valid_X, valid_y, n_epochs=1, batch_size=-1):
#     if batch_size == -1:
#         batch_size = train_X.shape[0]
#
#     model.fit(train_X, train_y, epochs=n_epochs, verbose=0, batch_size=batch_size)
#     acc = accuracy(valid_y, policy(model, valid_X, epsilon=1))
#     print("test accuracy = " + str(acc))

class dqn_network:
    def __init__(self, n_inputs, n_actions):
        # Have a state to observation mapping table
        # Have a state, action and value table
        # have a number of states parameters
        # have a number of possible actions
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.model = get_NN(n_inputs, n_actions) ## NN Model

    # def get_best_action_observation(self, obs):
    #     return get_best_action_observation(self.qtable, self.state_obs_list, obs)

    # def get_best_action_state(self, state):
    #     return get_best_action_state(self.qtable, state)

    # def get_n_states(self):
    #     return self.state_obs_list.shape[0]
    #
    # def get_n_actions(self):
    #     return self.n_actions

    # def fetch_q_table(self, qtable):
    #     return self.qtable

    # def update_q_table(self, qtable):
    #     if qtable.shape[0] != self.n_states or qtable.shape[1] != self.n_actions:
    #         print("Update q table: new q table\'s dimensions don\'t match existing table")
    #         exit(-1)
    #     else:
    #         self.qtable = qtable