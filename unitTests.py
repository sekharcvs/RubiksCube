import numpy as np
import cube

# Encode->Decode reversibility check
def checkEncodeDecodeSanity():
    diff = True
    for side in range(1,10):
        for a in range(3):
            for o in range(side):
                for d in range(2):
                    move = np.asarray([a, o, d]).reshape(1, 3).astype(np.int)
                    move_ = cube.decode_moves(cube.encode_moves(move, side), side)

                    if cube.check_exact_equal_arrays(move, move_) is False:
                        diff = False
                        break
    return diff

# Encode->Decode reversibility check
def checkMoveReversability():
    diff = True
    for side in range(1,10):
        for n_moves in range(1,20):
            C = cube.CubeObject(dim=side, n_moves=n_moves)
            C.apply_moves(cube.get_inverse_moves(C.moves_list))
            if cube.isSolved(C.state) is False:
                diff = False
                break
    return diff

def checkCheckEquivalentStates():
    # multiple rotations and reverse rotations lead to the same place
    n_moves = 25
    count = 0
    count_match = 0
    for side in range(1,10):
        for _ in range(100):
            C = cube.CubeObject(dim=side, n_moves=n_moves)
            state1 = C.state
            C = cube.CubeObject(dim=side, n_moves=n_moves)
            state2 = C.state

            count = count + 1
            if cube.check_equivalent_states(state1, state2) is True:
                count_match = count_match + 1

    percMatch = (100.0 * count_match)/count
    return percMatch, count_match, count

def checkRotations():
    # multiple rotations and reverse rotations lead to the same place
    n_moves = 10
    diff = True
    rotations = np.zeros(12).astype(np.int)
    rotations[0:4] = 0
    rotations[4:8] = 1
    rotations[8:12] = 2

    rotations_orig = np.repeat(rotations, 4)

    for side in range(1,10):
        for _ in range(10):
            rotations = np.random.permutation(rotations_orig)
            C = cube.CubeObject(dim=side, n_moves=n_moves)
            state1 = C.state
            for a in rotations:
                C.rotate_cube(a, 1)
                if cube.check_equivalent_states(state1, C.state) is False:
                    diff = False
                    break
    return diff


def checkRotationalMoveEquivalence():
    n_moves = 10
    diff = True

    for side in range(2,4):
        C = cube.CubeObject(side, n_moves=n_moves)
        state0 = C.state
        states_list = C.states_list
        moves_list = C.moves_list
        for iter in range(80):
            moves0 = cube.get_random_moves(side, n_moves)
            state1, _, _ = cube.moves_shuffle(state0, side, moves0, np.zeros([0, 3]).astype(np.int), np.zeros([0, 6, side, side]))
            moves1 = cube.get_random_moves(side, 1)
            move1 = moves1[0]

            a = np.zeros([0, 3]).astype(np.int)
            b = np.zeros([0, 6, side, side])
            state1_moved, _, _ = cube.moves_shuffle(state1, side, moves1, a, b)

            for a in range(3):
                for d in [-1, 1]:
                    for t in range(1, 4):
                        state2 = cube.rotate_cube(state1, a, d * t)
                        move2 = cube.get_move_axis_turns(move1, side, a, d * t)
                        moves2 = move2.reshape(1,3)
                        move1_ = cube.get_move_axis_turns(move1, side, a, 0)
                        if (move1 == move1_) is False:
                            _ = 1
                        state2_moved, _, _ = cube.moves_shuffle(state2, side, moves2, np.zeros([0, 3]).astype(np.int), np.zeros([0, 6, side, side]))

                        # state1 = np.load('temp1.npy')
                        # state2 = np.load('temp2.npy')

                        w = cube.check_equivalent_states(state1, state2)
                        x = cube.check_equivalent_states(state1_moved, state2_moved)
                        y = not cube.check_exact_equal_arrays(state1, state1_moved)
                        z = not cube.check_exact_equal_arrays(state2, state2_moved)
                        if (w and x and y and z) is False:
                            # np.save('temp1.npy', state1)
                            # np.save('temp2.npy', state2)
                            diff = False
                            break
    return diff

print("checkEncodeDecodeSanity: "+str(checkEncodeDecodeSanity()))
#print("checkMoveReversability: "+str(checkMoveReversability()))
#print("checkCheckEquivalentStates (equal_perc, numerator, denominator) (ideally equal_perc is 0.0): "+str(checkCheckEquivalentStates()))
#print("checkRotations: "+str(checkRotations()))
print("checkRotationalMoveEquivalence: "+str(checkRotationalMoveEquivalence()))
