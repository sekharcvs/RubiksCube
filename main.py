import cube
import numpy as np
from os.path import exists

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.utils import np_utils


# Common Constants
side = 2
n_moves = 2
num_classes = 3 * side * 2  # Maximum number of possible options per move. 3 Axes, side offsets, 2 directions

# Data Generation/Loading/Saving
LOAD_DATA = True
SAVE_DATA = False

if LOAD_DATA is False:
    # Generate Fresh Data
    iterations = 10000
    N = n_moves * iterations

    X = np.zeros([N, 6, side, side])
    y = np.zeros([N, 3])

    j = 0
    for i in range(iterations):
        C = cube.CubeObject(dim=side, n_moves=n_moves)
        moves_list = C.moves_list
        inverse_moves_list = cube.get_inverse_moves(moves_list)
        states_list = C.states_list

        for m in range(n_moves):
            X[j] = states_list[m]
            y[j] = inverse_moves_list[m]
            j = j+1

    # Groups equal states - Data clean-up step for better training
    X, idx, group_id = cube.group_equal_states(X)
    y = y[idx, :]

    max_group_id = group_id[-1]
    for id in range(max_group_id):
        first_idx = np.nonzero(group_id == id)[0][0]
        last_idx = np.nonzero(group_id == id)[0][-1]

        first_X = X[first_idx].copy()
        first_y = y[first_idx].copy()
        for idx in range(first_idx, last_idx+1):
            # For now removing confusions. Later, add all rotations of first_X
            # If equivalent states not have equivalent moves, pick the majority move and use that across all permutation
            X[idx] = first_X
            y[idx] = first_y

    # TODO: Add all rotations of the cube as training data
    idx = np.random.permutation(N)
    X = X[idx]
    y = y[idx, :]

    N_tr = np.int(np.floor(N * 0.7))
    N_va = np.int(np.floor(N * 0.15))
    N_te = N - N_tr - N_va

    train_X = X[0:N_tr]
    train_y = y[0:N_tr, :]

    valid_X = X[N_tr:N_tr+N_va]
    valid_y = y[N_tr:N_tr+N_va, :]

    test_X = X[N - N_te:N]
    test_y = y[N - N_te:N, :]

else:
    # Load Data from pre-saved files
    train_X = np.load("train_X.npy")
    train_y = np.load("train_y.npy")
    valid_X = np.load("valid_X.npy")
    valid_y = np.load("valid_y.npy")
    test_X = np.load("test_X.npy")
    test_y = np.load("test_y.npy")

if SAVE_DATA is True:
    np.save("train_X", train_X)
    np.save("train_y", train_y)
    np.save("valid_X", valid_X)
    np.save("valid_y", valid_y)
    np.save("test_X", test_X)
    np.save("test_y", test_y)

# Data Preparation
train_y = np_utils.to_categorical(cube.encode_moves(train_y, side), num_classes)
valid_y = np_utils.to_categorical(cube.encode_moves(valid_y, side), num_classes)
test_y = np_utils.to_categorical(cube.encode_moves(test_y, side), num_classes)


# Solver
def evaluate(model, test_X, num_classes):
    return np_utils.to_categorical(model.predict_classes(test_X), num_classes)


def accuracy(true, pred):
    a = np.argmax(true, axis=-1)
    b = np.argmax(pred, axis=-1)
    x = np.equal(a, b)
    return np.mean(x)


# Model Setup
LOAD_MODEL = False
SAVE_MODEL = True
UPDATE_MODEL = True

epochs = 1
dense_layer_size = 512
dropout = 0.2
num_classes = num_classes
model_name = "model.h5"

model = Sequential()
model.add(Flatten(input_shape=(6, side, side)))

# Fully connected layers
model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if LOAD_MODEL is True:
    if exists(model_name):
        model = load_model(model_name)

output_onehot = evaluate(model, valid_X, num_classes)
max_acc = accuracy(valid_y, output_onehot)
if UPDATE_MODEL is True:
    for count in range(epochs):
        model.fit(train_X, train_y, epochs=1, verbose=1)
        acc = accuracy(valid_y, evaluate(model, valid_X, num_classes))
        if acc > max_acc:
                print("max accuracy improved to "+str(acc)+"! Best model updated...")
                # save model and update best accuracy
                max_acc = acc
                if SAVE_MODEL is True:
                    model.save(model_name)


# Test on some real data
if SAVE_MODEL is True:
    # Get the best model
    model = load_model(model_name)

for i in range(20):
    C1 = cube.CubeObject(dim=side, n_moves=n_moves)
    cube.display(C1.state, C1.side, C1.colormap)
    j = 0

    while (cube.isSolved(C1.state) is False) and (j < 10):
        cube_state = np.asarray([C1.state]).astype(np.int)
        moves_encodings = model.predict_classes(cube_state)
        moves = cube.decode_moves(moves_encodings, side)
        C1.apply_moves(moves)
        cube.display(C1.state, C1.side, C1.colormap)
        j = j+1
