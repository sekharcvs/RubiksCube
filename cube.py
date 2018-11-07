import numpy as np
# from matplotlib import pyplot as plt
# import time
import matplotlib.pyplot as plt

FRONT = 1
LEFT = 0
RIGHT = 2
BACK = 3
TOP = 4
BOTTOM = 5

PI = 3.1415926
EPSILON = 1e-5
# Orange (0), Green (1), Red (2), Yellow (3), White (4), Blue (5)
default_map = np.array([[255, 165, 000], [000, 128, 000], [255, 000, 000], [255, 255, 000], [255, 255, 255], [000, 000, 255]]).astype(np.uint8)


def encode_moves(moves, side):
    N = moves.shape[0]
    move_encodings = np.zeros([N, 1])
    for i in range(N):
        a = moves[i, 0]
        o = moves[i, 1]
        d = moves[i, 2]
        move_encodings[i] = a * side * 2 + o * 2 + d

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

        moves[i] = np.asarray([a, o, d]).reshape(1, 3).astype(np.int)
    return moves


def isSolved(state):
    # returns whether the cube is solved
    solved = True
    for i in range(6):
        face = state[i, :, :]
        val = face[0, 0]
        diff = face - val
        if np.sum(np.fabs(diff)) != 0:
            solved = False
    return solved

def check_exact_equal_arrays(arr1, arr2):
    check = False
    if np.sum(np.absolute(arr1 - arr2)) < EPSILON:
        check = True
    return check

def check_equivalent_states(state1, state2):
    for idx1 in range(6):
        flag = False
        face1 = state1[idx1].astype(np.int)
        for idx2 in range(6):
            face2a = state2[idx2].astype(np.int)
            face2b = face2a[:, :].transpose()
            face2c = face2a[::-1, ::-1]
            face2d = face2a[::-1, ::-1].transpose()
            face2e = face2a[::-1, :]
            face2f = face2a[::-1, :].transpose()
            face2g = face2a[:, ::-1]
            face2h = face2a[:, ::-1].transpose()
            if np.array_equal(face1, face2a) or np.array_equal(face1, face2b) or np.array_equal(face1, face2c) or np.array_equal(face1, face2d) or np.array_equal(face1, face2e) or np.array_equal(face1, face2f) or np.array_equal(face1, face2g) or np.array_equal(face1, face2h):
                flag = True
                break

        if flag is False:
            return False, np.asarray((0, 0, 0))

    # tempStateX = state1.copy()
    # for xturns in range(3):
    #     tempStateX = rotate_turn_counterclockwise(tempStateX, 0)
    #     if check_exact_equal_arrays(tempStateX, state2):
    #         return True, np.asarray((xturns, 0, 0))
    #     tempStateY = tempStateX.copy()
    #     for yturns in range(3):
    #         tempStateY = rotate_turn_counterclockwise(tempStateY, 1)
    #         if check_exact_equal_arrays(tempStateY, state2):
    #             return True, np.asarray((xturns, yturns, 0))
    #         tempStateZ = tempStateY.copy()
    #         for zturns in range(3):
    #             tempStateZ = rotate_turn_counterclockwise(tempStateZ, 2)
    #             if check_exact_equal_arrays(tempStateZ, state2):
    #                 return True, np.asarray((xturns, yturns, zturns))

    tempState = state1.copy()
    for turns in range(4):
        tempState = rotate_cube(tempState, 0, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True

    tempState = state1.copy()
    tempState = rotate_cube(tempState, 1, 1)
    for turns in range(4):
        tempState = rotate_cube(tempState, 2, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True

    tempState = state1.copy()
    tempState = rotate_cube(tempState, 1, 2)
    for turns in range(4):
        tempState = rotate_cube(tempState, 0, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True

    tempState = state1.copy()
    tempState = rotate_cube(tempState, 1, 3)
    for turns in range(4):
        tempState = rotate_cube(tempState, 2, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True

    tempState = state1.copy()
    tempState = rotate_cube(tempState, 2, 1)
    for turns in range(4):
        tempState = rotate_cube(tempState, 1, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True

    tempState = state1.copy()
    tempState = rotate_cube(tempState, 2, 3)
    for turns in range(4):
        tempState = rotate_cube(tempState, 1, turns)
        if check_exact_equal_arrays(tempState, state2):
            return True


    #return False, np.asarray((0, 0, 0))
    return False


def get_move_axis_turns(move, side, axis, n_turns):
    # If a cube state2 is the same as state1 but with rotation of cube according to
    # facing the axis and performing n_turns number of anti-clockwise turns
    # equivalent needed move corresponding to input move on state2 is returned
    # n_turns can be either positive or negative
    a = np.int(move[0])
    o = np.int(move[1])
    d = np.int(move[2])

    side = np.int(side)
    axis = np.int(axis)
    n_turns = np.int(n_turns)

    c = np.round(np.cos(PI * n_turns / 2))
    s = np.round(np.sin(PI * n_turns / 2))

    corr = np.zeros(3)

    if axis == 0:
        # X-Axis turns
        x = np.array([0, 0])
        y = np.array([0, -1])
        z = np.array([-1, 0])
    elif axis == 1:
        # Y-Axis turns
        x = np.array([1, 0])
        y = np.array([0, 0])
        z = np.array([0, 1])
    else:  # axis == 2
        # Z-Axis turns
        x = np.array([1,0])
        y = np.array([0,-1])
        z = np.array([0,0])

    if a == 0:
        v = x
    elif a == 1:
        v = y
    else: # a == 2
        v = z

    v = v.reshape([2,1])
    R = np.array([[c, -s], [s, c]])
    v1 = np.matmul(R, v).reshape(2)

    if a == axis:
        corr[:] = 0
        corr[a] = 1
    else:
        corr[0] = np.dot(v1, x)
        corr[1] = np.dot(v1, y)
        corr[2] = np.dot(v1, z)

    # decide final new axis, offset and direction
    if np.abs(corr[0]) > EPSILON:
        a1 = 0
        c = corr[0]
    elif (np.abs(corr[1]) > EPSILON):
        a1 = 1
        c = corr[1]
    else:
        a1 = 2
        c = corr[2]

    # decide on direction and offset
    if c >= 0:
        o1 = o
        d1 = d
    else:
        o1 = side - 1 - o
        d1 = 1 - d
    move_equivalent = np.asarray([a1, o1, d1]).astype(np.int)

    return move_equivalent


# Low complexity version to group all the equivalent states together
def group_equal_states(states_list):
    n_states = states_list.shape[0]
    lla = [[states_list[0]]]
    lli = [[0]]
    ngroups = 1
    # times = np.zeros([n_states, 1])
    # times[0] = time.time()
    for c in range(1, n_states):
        tempState = states_list[c]
        group_found = False
        for gr in range(ngroups):
            refState = (lla[gr])[0]
            if check_equivalent_states(refState, tempState) is True:
                lla[gr].append(tempState)
                lli[gr].append(c)
                group_found = True
                break
        if group_found is False:
            ngroups = ngroups+1
            lla.append([tempState])
            lli.append([c])
        # times[c] = time.time()
    # plt.plot(times - times[0])
    temp_states_list = states_list.copy()
    idx = np.asarray(range(n_states))
    group_id = np.zeros([n_states,1]).reshape([-1]).astype(np.int)
    c = 0
    for gr in range(ngroups):
        for i in range(len(lla[gr])):
            temp_states_list[c] = (lla[gr])[i]
            idx[c] = (lli[gr])[i]
            group_id[c] = gr
            # times[c] = time.time()
            c = c+1
    # plt.plot(times - times[0])

    states_list = temp_states_list
    return states_list, idx, group_id


def display(state, side, colormap=default_map):
    x = np.zeros([side * 3, side * 4, 3]).astype(np.uint8)
    x[:, :, :] = 200

    for i in range(6):
        if i < 4:
            # Sides
            startx = i * side
            starty = side
            for j in range(starty, starty + side):
                for k in range(startx, startx + side):
                    x[j, k, :] = colormap[state[i, j - starty, k - startx]]

        if i == 4:
            # Top
            startx = side
            starty = 0
            for j in range(starty, starty + side):
                for k in range(startx, startx + side):
                    x[j, k, :] = colormap[state[i, j - starty, k - startx]]

        if i == 5:
            # Bottom
            startx = side
            starty = 2 * side
            for j in range(starty, starty + side):
                for k in range(startx, startx + side):
                    x[j, k, :] = colormap[state[i, j - starty, k - startx]]

    plt.imshow(x)
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, 4 * side, 1))
    ax.set_yticks(np.arange(-.5, 3 * side, 1))
    ax.grid(color='k', linestyle='-', linewidth=2)

    return


def get_inverse_moves(moves):
    n_moves = moves.shape[0]
    inverse_moves = moves[::-1, :].copy()
    for i in range(n_moves):
        inverse_moves[i, 0:2] = moves[n_moves - 1 - i, 0:2]
        inverse_moves[i, 2] = 1 - moves[n_moves - 1 - i, 2]

    return inverse_moves


def rotate_turn_counterclockwise(state, axis):
    temp = np.zeros(state.shape).astype(np.int)
    if axis == 1:
        # Y-Axis
        temp[LEFT, :, :] = state[BACK, :, :]
        temp[FRONT, :, :] = state[LEFT, :, :]
        temp[RIGHT, :, :] = state[FRONT, :, :]
        temp[BACK, :, :] = state[RIGHT, :, :]

        temp[TOP, :, :] = np.rot90(state[TOP, :, :], 1)
        temp[BOTTOM, :, :] = np.rot90(state[BOTTOM, :, :], 3)
    elif axis == 0:
        # X-Axis
        temp[BACK, :, :] = state[TOP, ::-1, ::-1]
        temp[TOP, :, :] = state[FRONT, :, :]
        temp[FRONT, :, :] = state[BOTTOM, :, :]
        temp[BOTTOM, :, :] = state[BACK, ::-1, ::-1]

        temp[LEFT, :, :] = np.rot90(state[LEFT, :, :], 1)
        temp[RIGHT, :, :] = np.rot90(state[RIGHT, :, :], 3)
    else:
        # Z-Axis
        temp[LEFT, :, :] = np.rot90(state[TOP, :, :], 1)
        temp[TOP, :, :] = np.rot90(state[RIGHT, :, :], 1)
        temp[RIGHT, :, :] = np.rot90(state[BOTTOM, :, :], 1)
        temp[BOTTOM, :, :] = np.rot90(state[LEFT, :, :], 1)

        temp[FRONT, :, :] = np.rot90(state[FRONT, :, :], 1)
        temp[BACK, :, :] = np.rot90(state[BACK, :, :], 3)

    state = temp
    return state


def rotate_cube(state, axis, turns):
    turns = (np.arange(0, 4))[np.remainder(turns, 4)]
    for i in range(turns):
        state = rotate_turn_counterclockwise(state, axis)
    return state


def update_moves_states(state, moves_list, states_list, axis, offset, direction):
    temp = [np.asarray([axis,offset,direction])]
    moves_list = np.append(moves_list, temp, axis=0)
    states_list = np.append(states_list, [state], axis=0)
    return moves_list, states_list


def moveY(state, side, offset, direction):
    # DO NOT CALL THIS FUNCTION FROM ANYWHERE BUT THE move() ROUTINE!!!!!
    # Does only a 90 degree rotation along Y axis only
    # Direction - 0 (Clockwise when looking along that axes), 1 (Anti-Clockwise)
    # Standard Right hand axes convention. X (-->) Y (V - downward) Z (x - Into the plane).
    # For any face, (0,0) is always the top left according to the figure in:
    # https://ruwix.com/the-rubiks-cube/japanese-western-color-schemes/
    # Offset is the offset from the face that comes first along that axis - ex: offset 0 for Y Axis is TOP

    # Faces
    if offset == 0 or offset == side - 1:
        # Along with sides, even face needs rotation
        if offset == 0:
            face = TOP
            f = state[face, :, :]
            if direction == 1:
                f1 = np.rot90(f, 1)  # 90 degrees anti-clockwise
            else:
                f1 = np.rot90(f, 3)  # 270 degrees anti-clockwise
        else:
            face = BOTTOM
            f = state[face, :, :]
            if direction == 1:
                f1 = np.rot90(f, 3)  # 270 degrees anti-clockwise
            else:
                f1 = np.rot90(f, 1)  # 90 degrees anti-clockwise
        state[face, :, :] = f1

    # Sides
    a = state[FRONT, offset, :].copy()
    b = state[RIGHT, offset, :].copy()
    c = state[BACK,  offset, :].copy()
    d = state[LEFT,  offset, :].copy()

    if direction == 1:
        # Anti-clockwise
        state[FRONT, offset, :] = d
        state[RIGHT, offset, :] = a
        state[BACK,  offset, :] = b
        state[LEFT,  offset, :] = c
    else:
        # Positive
        state[FRONT, offset, :] = b
        state[RIGHT, offset, :] = c
        state[BACK,  offset, :] = d
        state[LEFT,  offset, :] = a

    return state


def move(state, moves_list, states_list, side, axis, offset, direction):
    # Does only a 90 degree rotation
    # Direction - 0 (Clockwise when looking along that axes), 1 (Anti-Clockwise)
    # Standard Right hand axes convention. X (-->) Y (V - downward) Z (x - Into the plane).
    # For any face, (0,0) is always the top left according to the figure in:
    # https://ruwix.com/the-rubiks-cube/japanese-western-color-schemes/
    # Offset is the offset from the face that comes first along that axis - ex: offset 0 for Y Axis is TOP
    # Updates the moves buffer and cube_states

    # Use Y axis rotation and cube rotations as basis
    if axis == 1:
        # Y-Axis
        state = moveY(state, side, offset, direction)
    if axis == 0:
        # X-Axis
        state = rotate_cube(state, 1, 1)
        state = rotate_cube(state, 0, 1)
        state = moveY(state, side, offset, direction)
        state = rotate_cube(state, 0, -1)
        state = rotate_cube(state, 1, -1)
    if axis == 2:
        # Z-Axis
        state = rotate_cube(state, 0, 1)
        state = moveY(state, side, offset, direction)
        state = rotate_cube(state, 0, -1)


    moves_list, states_list = update_moves_states(state, moves_list, states_list, axis, offset, direction)

    return state, moves_list, states_list


def moves_shuffle(state, side, moves, moves_list, states_list):
    n_moves = moves.shape[0]
    for i in range(n_moves):
        axis = moves[i, 0].astype(np.int)
        offset = moves[i, 1].astype(np.int)
        direction = moves[i, 2].astype(np.int)
        state_ = state.copy()
        state, moves_list, states_list = move(state_, moves_list, states_list, side, axis, offset, direction)
    return state, moves_list, states_list

def get_random_moves(side, n_moves=1):
    moves = np.zeros([n_moves, 3]).astype(np.int)
    for i in range(n_moves):
        axis = np.random.randint(0, 3)
        offset = np.random.randint(0, side)
        direction = np.random.randint(0, 2)
        moves[i, :] = [axis, offset, direction]
    return moves

def random_shuffle(state, side, n_moves, moves_list, states_list):
    moves = get_random_moves(side, n_moves)
    state, moves_list, states_list = moves_shuffle(state, side, moves, moves_list, states_list)
    return state, moves_list, states_list


class CubeObject:
    def __init__(self, dim, n_moves=100):
        self.side = dim
        self.area = self.side * self.side
        # Order: Left,Front,Right,Back,Up,Down - [0,1,2,3,4,5] - [Orange (0), Green (1), Red (2), Yellow (3), White (4), Blue (5)]
        self.state = np.zeros([6, self.side, self.side]).astype(np.int)
        self.colormap = default_map
        self.moves_list = np.zeros([0, 3]).astype(np.int)
        self.states_list = np.zeros([0, 6, self.side, self.side])

        for i in range(6):
            # Initialize a solved cube
            # Top face is 0, side faces are 1-4
            # Bottom face is 5
            self.state[i, :, :] = i

        self.state, self.moves_list, self.states_list = random_shuffle(self.state, self.side, n_moves, self.moves_list, self.states_list)

    def update_lists(self, moves_list, states_list):
        self.moves_list = moves_list
        self.states_list = states_list

    def apply_moves(self, moves):
        self.state, self.moves_list, self.states_list = moves_shuffle(self.state, self.side, moves, self.moves_list, self.states_list)

    def rotate_cube(self, axis, turns):
        self.state = rotate_cube(self.state, axis, turns)

    def set_state(self, state):
        self.state = state
        self.moves_list = np.zeros([0, 3]).astype(np.int)
        self.states_list = np.zeros([0, 6, self.side, self.side])


