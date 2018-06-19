import cube
import numpy as np
from os.path import exists

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.utils import np_utils


def encode_moves(moves, side):
    N = moves.shape[0]
    move_encodings = np.zeros([N,1])
    for i in range(N):
        a = moves[i, 0]
        o = moves[i, 1]
        d = moves[i, 2]
        move_encodings[i] = a * (side * 2) + o * (2) + d

    return move_encodings


def decode_moves(move_encodings, side):
    N = move_encodings.shape[0]

    M = 3*side*2

    moves = np.zeros([N, 3]).astype(np.int)
    for i in range(N):
        n = move_encodings[i] % M

        d = n % 2
        o = np.floor(n/2).astype(np.int) % side
        a = np.floor(n/(2 * side)).astype(np.int) % 3

        moves[i] = np.asarray([a, o, d]).reshape(1,1,3).astype(np.int)
    return moves


# Common Constants
side = 2
n_moves = 2
num_classes = 3 * side * 2  # Maximum number of possible options per move. 3 Axes, side offsets, 2 directions

# Data Generation/Loading/Saving
LOAD_DATA = False
SAVE_DATA = True

if LOAD_DATA is False:
    # Generate Fresh Data
    iterations = 100000
    N = n_moves * iterations

    X = np.zeros([N, 6, side, side])
    y = np.zeros([N, 3])

    j = 0
    for i in range(iterations):
        C = cube.cube(dim=side, n_moves=n_moves)
        moves = C.moves
        inverse_moves = C.get_inverse_moves(moves)
        cube_states = C.cube_states

        for m in range(n_moves):
            X[j] = cube_states[m]
            y[j] = inverse_moves[m]
            j = j+1

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
train_y = np_utils.to_categorical(encode_moves(train_y, side), num_classes)
valid_y = np_utils.to_categorical(encode_moves(valid_y, side), num_classes)
test_y = np_utils.to_categorical(encode_moves(test_y, side), num_classes)


# Solver

def evaluate(model, test_X, num_classes):
    return np_utils.to_categorical(model.predict_classes(test_X), num_classes)


def accuracy(true, pred):
    a = np.argmax(true, axis = -1)
    b = np.argmax(pred, axis = -1)
    x = np.equal(a, b)
    return np.mean(x)


# Model Setup
LOAD_MODEL = True
SAVE_MODEL = True
UPDATE_MODEL = True

epochs = 5
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
    C1 = cube.cube(dim=side, n_moves=n_moves)
    C1.display()
    j = 0

    while (C1.isSolved() is False) and (j < 10):
        cube_state = np.asarray([C1.cube]).astype(np.int)
        moves_encodings = model.predict_classes(cube_state)
        moves = decode_moves(moves_encodings, side)
        C1.moves_shuffle(moves)
        C1.display()
        j = j + 1