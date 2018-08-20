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


print("checkEncodeDecodeSanity: "+str(checkEncodeDecodeSanity()))
print("checkMoveReversability: "+str(checkMoveReversability()))
print("checkCheckEquivalentStates (equal_perc, numerator, denominator) (ideally equal_perc is 0.0): "+str(checkCheckEquivalentStates()))
print("checkRotations: "+str(checkRotations()))
