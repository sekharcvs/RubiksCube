import cube
import numpy as np
from os.path import exists

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.utils import np_utils

side = 2
n_moves = 2
iterations = 2

N = n_moves * iterations

X = np.zeros([N, 6 * side * side])
y = np.zeros([N, 3])

j = 0
for i in range(iterations):
    C = cube.cube(dim=side, n_moves=n_moves)
    moves = C.moves
    inverse_moves = C.get_inverse_moves(moves)
    cube_states = C.cube_states.reshape(n_moves, -1)

    for m in range(n_moves):
        X[j] = cube_states[m]
        y[j] = inverse_moves[m]
        j = j+1

idx = np.random.permutation(N)
X = X[idx, :]
y = y[idx, :]
y_axis = np_utils.to_categorical(y[:,0], 3)
y_offset = np_utils.to_categorical(y[:,1], side)
y_direction = np_utils.to_categorical(y[:,2], 2)

N_tr = np.int(np.floor(N * 0.7))
N_va = np.int(np.floor(N * 0.15))
N_te = N - N_tr - N_va

train_X = X[0:N_tr, :]
train_y = y[0:N_tr, :]
train_y_axis = y_axis[0:N_tr, :]
train_y_offset = y_offset[0:N_tr, :]
train_y_direction = y_direction[0:N_tr, :]

valid_X = X[N_tr:N_tr+N_va, :]
valid_y = y[N_tr:N_tr+N_va, :]
valid_y_axis = y_axis[N_tr:N_tr+N_va, :]
valid_y_offset = y_offset[N_tr:N_tr+N_va, :]
valid_y_direction = y_direction[N_tr:N_tr+N_va, :]

test_X = X[N - N_te:N, :]
test_y = y[N - N_te:N, :]
test_y_axis = y_axis[N - N_te:N, :]
test_y_offset = y_offset[N - N_te:N, :]
test_y_direction = y_direction[N - N_te:N, :]

# Solver
LOAD_MODEL = False
SAVE_MODEL = True
UPDATE_MODEL = True


def evaluate(model, test_X, nc):
    return np_utils.to_categorical(model.predict_classes(test_X), nc)


def accuracy(true, pred):
    a = np.argmax(true, axis = -1)
    b = np.argmax(pred, axis = -1)
    x = np.equal(a, b)
    return np.mean(x)

# TODO: Convert to multi-task learning model later
# Axis
epochs = 10
dense_layer_size = 512
dropout = 0.2
num_classes = 3 # X, Y, Z
model_name = "axis.h5"

model_axis = Sequential()

# Fully connected layers
model_axis.add(Dense(dense_layer_size, input_dim=6 * side * side, activation='relu'))
model_axis.add(Dropout(dropout))

model_axis.add(Dense(dense_layer_size, activation='relu'))
model_axis.add(Dropout(dropout))

model_axis.add(Dense(num_classes, activation='softmax'))

model_axis.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if LOAD_MODEL is True:
    if exists(model_name):
        model_axis = load_model(model_name)

acc = evaluate(model_axis, valid_X, num_classes)
max_acc = accuracy(valid_y_axis, acc)
if UPDATE_MODEL is True:
    for count in range(epochs):
        model_axis.fit(train_X, train_y_axis, epochs=1, verbose=1)
        acc = accuracy(valid_y_axis, evaluate(model_axis, valid_X, num_classes))
        if acc > max_acc:
                print("max accuracy improved to "+str(acc)+"! Best model updated...")
                # save model and update best accuracy
                max_acc = acc
                if SAVE_MODEL is True:
                    model_axis.save(model_name)


# Offset
epochs = 10
dense_layer_size = 512
dropout = 0.2
num_classes = side
model_name = "offset.h5"

model_offset = Sequential()

# Fully connected layers
model_offset.add(Dense(dense_layer_size, input_dim=6 * side * side, activation='relu'))
model_offset.add(Dropout(dropout))

model_offset.add(Dense(dense_layer_size, activation='relu'))
model_offset.add(Dropout(dropout))

model_offset.add(Dense(num_classes, activation='softmax'))

model_offset.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if LOAD_MODEL is True:
    if exists(model_name):
        model_offset = load_model(model_name)

acc = evaluate(model_offset, valid_X, num_classes)
max_acc = accuracy(valid_y_offset, acc)
if UPDATE_MODEL is True:
    for count in range(epochs):
        model_offset.fit(train_X, train_y_offset, epochs=1, verbose=1)
        acc = accuracy(valid_y_offset, evaluate(model_offset, valid_X, num_classes))
        if acc > max_acc:
                print("max accuracy improved to "+str(acc)+"! Best model updated...")
                # save model and update best accuracy
                max_acc = acc
                if SAVE_MODEL is True:
                    model_offset.save(model_name)


# Direction
epochs = 10
dense_layer_size = 512
dropout = 0.2
num_classes = 2
model_name = "direction.h5"

model_direction = Sequential()

# Fully connected layers
model_direction.add(Dense(dense_layer_size, input_dim=6 * side * side, activation='relu'))
model_direction.add(Dropout(dropout))

model_direction.add(Dense(dense_layer_size, activation='relu'))
model_direction.add(Dropout(dropout))

model_direction.add(Dense(num_classes, activation='softmax'))

model_direction.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if LOAD_MODEL is True:
    if exists(model_name):
        model_direction = load_model(model_name)

acc = evaluate(model_direction, valid_X, num_classes)
max_acc = accuracy(valid_y_direction, acc)
if UPDATE_MODEL is True:
    for count in range(epochs):
        model_direction.fit(train_X, train_y_direction, epochs=1, verbose=1)
        acc = accuracy(valid_y_direction, evaluate(model_direction, valid_X, num_classes))
        if acc > max_acc:
                print("max accuracy improved to "+str(acc)+"! Best model updated...")
                # save model and update best accuracy
                max_acc = acc
                if SAVE_MODEL is True:
                    model_direction.save(model_name)


# TODO: Test on real rubik's cube and implement all the motions one after another
for i in range(20):
    C1 = cube.cube(dim=side, n_moves=1)
    C1.display()

    j = 0
    while (C1.isSolved() is False) and (j < 10):
        cube1 = C1.cube.reshape(1, -1)
        a = model_axis.predict_classes(cube1)
        o = model_offset.predict_classes(cube1)
        d = model_direction.predict_classes(cube1)
        C1.moves_shuffle(np.asarray([a,o,d]).transpose())
        C1.display()
        j = j + 1
