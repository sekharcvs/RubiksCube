import cube
import numpy as np
from os.path import exists

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.utils import np_utils


# Common Constants
side = 2
n_moves = 1
num_classes = 3 * side * 2  # Maximum number of possible options per move. 3 Axes, side offsets, 2 directions

# Data Generation/Loading/Saving
LOAD_DATA = False
SAVE_DATA = True

if LOAD_DATA is False:
    # Generate Fresh Data
    iterations = 1000
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

    # Prune out redundant states, but cover all possible rotations of unique states
    max_group_id = group_id[-1]
    n_rotations = 1 + 3 * 2 * 3  # Possible unique rotations of the cube including original state
    N1 = max_group_id * n_rotations
    X1 = np.zeros([N1, 6, side, side])
    y1 = np.zeros([N1, 3])

    X2 = X1.copy()
    y2 = y1.copy()

    X3 = X1.copy()
    y3 = y1.copy()

    idx = 0
    for id in range(max_group_id):
        first_idx = np.nonzero(group_id == id)[0][0]

        first_X = X[first_idx].copy()
        first_y = y[first_idx].copy()

        X1[idx] = first_X
        y1[idx] = first_y

        # TODO: Remove this later. Have unit tests for move(), rotate(), check_equal_states() and get_move_axis_turns()
        first_X = np.asarray([[[0., 0.],  [0., 0.]], [[1., 5.],  [1., 5.]], [[2., 2.],  [2., 2.]], [[4., 3.],  [4., 3.]], [[4., 1.], [4., 1.]], [[5., 3.], [5., 3.]]])
        first_y = np.asarray([2., 1., 1.])

        TEMP = np.zeros([0, 3]).astype(np.int)
        TEMP1 = np.zeros([0, 6, side, side]).astype(np.int)
        X2[idx] = cube.move(first_X.astype(np.int), TEMP, TEMP1, side, first_y[0].astype(np.int), first_y[1].astype(np.int), first_y[2].astype(np.int))[0]
        first_X_moved = X2[idx].copy()

        idx = idx + 1
        for a in range(3):
            for d in [-1, 1]:
                for t in range(1,4):
                    # For now removing confusions. Later, add all rotations of first_X
                    # If equivalent states not have equivalent moves, pick the majority move and use that across all permutation
                    X1[idx] = cube.rotate_cube(first_X, a, d * t)
                    y1[idx] = cube.get_move_axis_turns(first_y, a, d * t)

                    X2[idx] = first_X_moved
                    X3[idx] = cube.move(X1[idx].astype(np.int), TEMP, TEMP1, side, y1[idx][0].astype(np.int), y1[idx][1].astype(np.int), y1[idx][2].astype(np.int))[0]
                    X3[idx] = cube.rotate_cube(X3[idx], a, -d * t)
                    idx = idx + 1

    N = N1
    X = X1
    y = y1

    # Random permutations to shuffle data
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

epochs = 10
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
