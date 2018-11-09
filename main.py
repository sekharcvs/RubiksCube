import cube
import numpy as np
from os.path import exists

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.utils import np_utils



# Common Constants
side = 2
n_moves_max = 6
num_classes = 3 * side * 2  # Maximum number of possible options per move. 3 Axes, side offsets, 2 directions

# Data Generation/Loading/Saving
LOAD_DATA = True # Warning: Because grouping is ideally done on full states as smaller moved states are resolved first for avoiding cycles., Load + Update is not ideal
UPDATE_DATA = False
SAVE_DATA = False


if LOAD_DATA is True:
    # Load Data from pre-saved files
    train_X = np.load("train_X_{}.npy".format(n_moves_max))
    train_y = np.load("train_y_{}.npy".format(n_moves_max))
    valid_X = np.load("valid_X_{}.npy".format(n_moves_max))
    valid_y = np.load("valid_y_{}.npy".format(n_moves_max))
else:
    # Load Data from pre-saved files
    train_X = np.zeros([0, 6, side, side])
    train_y = np.zeros([0, 3])
    valid_X = np.zeros([0, 6, side, side])
    valid_y = np.zeros([0, 3])

if UPDATE_DATA is True:
    # Generate Fresh Data
    iterations = np.zeros(n_moves_max + 1)
    N = 0
    n_moves_start = 1
    n_moves_end = n_moves_max + 1
    for n_moves in range(n_moves_start, n_moves_end):
        iterations[n_moves] = np.int(n_moves * n_moves * 2000)
        # iterations[n_moves_max] = 10
        N = N + n_moves * iterations[n_moves]

    N = int(N)
    X = np.zeros([N, 6, side, side])
    y = np.zeros([N, 3])


    j = 0
    for n_moves in range(n_moves_start, n_moves_end):
        print("Creating Random Moves: "+str(n_moves))
        for i in range(int(iterations[n_moves])):
            C = cube.CubeObject(dim=side, n_moves=n_moves)
            moves_list = C.moves_list
            inverse_moves_list = cube.get_inverse_moves(moves_list)
            states_list = C.states_list

            for m in range(n_moves):
                X[j] = states_list[n_moves - 1 - m]
                y[j] = inverse_moves_list[m]
                j = j+1

    # Groups equal states - Data clean-up step for better training
    X, idx, group_id = cube.group_equal_states(X)
    y = y[idx, :]

    # Prune out redundant states, but cover all possible rotations of unique states
    max_group_id = group_id[-1]
    n_rotations = 4 * 6  # Possible unique rotations of the cube including original state
    N1 = max_group_id * n_rotations
    X1 = np.zeros([N1, 6, side, side])
    y1 = np.zeros([N1, 3])

    idx = 0
    for id in range(max_group_id):
        print("Generating Equivalent States: "+str(id)+" out of "+str(max_group_id)+" groups")
        first_idx = np.nonzero(group_id == id)[0][0]

        first_X = X[first_idx].copy()
        first_y = y[first_idx].copy()

        # np.save('first_y.npy', first_y)
        # first_y = np.load('first_y.npy')

        tempState = first_X.copy()
        tempMove  = first_y.copy()
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 0, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 0, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1

        tempState = first_X.copy()
        tempMove  = first_y.copy()
        tempState = cube.rotate_cube(tempState, 1, 1)
        tempMove = cube.get_move_axis_turns(tempMove, side, 1, 1)
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 2, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 2, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1


        tempState = first_X.copy()
        tempMove  = first_y.copy()
        tempState = cube.rotate_cube(tempState, 1, 2)
        tempMove = cube.get_move_axis_turns(tempMove, side, 1, 2)
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 0, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 0, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1


        tempState = first_X.copy()
        tempMove  = first_y.copy()
        tempState = cube.rotate_cube(tempState, 1, 3)
        tempMove = cube.get_move_axis_turns(tempMove, side, 1, 3)
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 2, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 2, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1


        tempState = first_X.copy()
        tempMove  = first_y.copy()
        tempState = cube.rotate_cube(tempState, 2, 1)
        tempMove = cube.get_move_axis_turns(tempMove, side, 2, 1)
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 1, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 1, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1


        tempState = first_X.copy()
        tempMove  = first_y.copy()
        tempState = cube.rotate_cube(tempState, 2, 3)
        tempMove = cube.get_move_axis_turns(tempMove, side, 2, 3)
        for turns in range(4):
            tempState = cube.rotate_cube(tempState, 1, turns)
            tempMove = cube.get_move_axis_turns(tempMove, side, 1, turns)
            X1[idx] = tempState
            y1[idx] = tempMove
            idx = idx + 1


        # for a in range(3):
        #     for d in [-1, 1]:
        #         for t in range(1,4):
        #             # For now removing confusions. Later, add all rotations of first_X
        #             # If equivalent states not have equivalent moves, pick the majority move and use that across all permutation
        #             X1[idx] = cube.rotate_cube(first_X, a, d * t)
        #             y1[idx] = cube.get_move_axis_turns(first_y, a, d * t)
        #             idx = idx + 1

    N = N1
    X = X1
    y = y1

    # Random permutations to shuffle data
    idx = np.random.permutation(N)
    X = X[idx]
    y = y[idx, :]

    N_tr = np.int(np.floor(N * 0.85))
    N_va = N - N_tr

    train_X = np.append(train_X, X[0:N_tr], axis=0)
    train_y = np.append(train_y, y[0:N_tr, :], axis=0)

    valid_X = np.append(valid_X, X[N_tr:N], axis=0)
    valid_y = np.append(valid_y, y[N_tr:N, :], axis=0)

if SAVE_DATA is True:
    np.save("train_X_{}".format(n_moves_max), train_X)
    np.save("train_y_{}".format(n_moves_max), train_y)
    np.save("valid_X_{}".format(n_moves_max), valid_X)
    np.save("valid_y_{}".format(n_moves_max), valid_y)

# Data Preparation
train_y = np_utils.to_categorical(cube.encode_moves(train_y, side), num_classes)
valid_y = np_utils.to_categorical(cube.encode_moves(valid_y, side), num_classes)

# Solver
def evaluate(model, test_X, num_classes):
    return np_utils.to_categorical(model.predict_classes(test_X), num_classes)


def accuracy(true, pred):
    a = np.argmax(true, axis=-1)
    b = np.argmax(pred, axis=-1)
    x = np.equal(a, b)
    return np.mean(x)


# Model Setup
LOAD_MODEL = True
SAVE_MODEL = True
UPDATE_MODEL = True

epochs = 200
dense_layer_size = 300
dropout = 0.0
num_classes = num_classes
load_model_name = "model_{}moves.h5".format(n_moves_max)
save_model_name = "model_{}moves.h5".format(n_moves_max)

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
    if exists(load_model_name):
        model = load_model(load_model_name)

output_onehot = evaluate(model, valid_X, num_classes)
max_acc = accuracy(valid_y, output_onehot)
print("Epoch 0 acc "+str(max_acc))
if UPDATE_MODEL is True:
    for count in range(epochs):
        model.fit(train_X, train_y, epochs=1, verbose=1)
        acc = accuracy(valid_y, evaluate(model, valid_X, num_classes))
        print("Epoch "+str(count)+" acc "+str(acc))
        if acc >= max_acc or True: ########## Always updating model ##########
                print("max accuracy improved to "+str(acc)+"! Best model updated...")
                # save model and update best accuracy
                max_acc = acc
                if SAVE_MODEL is True:
                    model.save(save_model_name)


# Test on some real data
if SAVE_MODEL is True:
    # Get the best model
    model = load_model(save_model_name)


# Evaluation
n_moves_low = 1
n_moves_high = n_moves_max + 1

total_count = np.zeros([n_moves_high])
solved_count = np.zeros([n_moves_high])

for i in range(5000):
    n_moves = np.random.randint(n_moves_low, n_moves_high)

    C1 = cube.CubeObject(dim=side, n_moves=n_moves)
    #cube.display(C1.state, C1.side, C1.colormap)

    total_count[n_moves] = total_count[n_moves] + 1

    j = 0
    while (cube.isSolved(C1.state) is False) and (j < 21):
        cube_state = np.asarray([C1.state]).astype(np.int)
        moves_encodings = model.predict_classes(cube_state)
        moves = cube.decode_moves(moves_encodings, side)
        C1.apply_moves(moves)
        #cube.display(C1.state, C1.side, C1.colormap)
        j = j+1

    if cube.isSolved(C1.state) is True:
        solved_count[n_moves] = solved_count[n_moves] + 1

for i in range(1, n_moves_high):
    percentage_solved = (100.0 * solved_count[i]) / total_count[i]
    print("Results:\nPercentage Solved "+str(i)+" moves away: "+str(percentage_solved))

percentage_solved = (100.0 * sum(solved_count)) / sum(total_count)
print("Results:\nPercentage Solved overall: "+str(percentage_solved))
