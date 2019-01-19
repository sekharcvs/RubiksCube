import numpy as np
import cube
import dqn_network_solver as solver
import dqn_network as network
import sys

N_MOVES_AWAY_START_MAX = 2
SIDE = 2
N_MOVES_SOLVE_MAX = N_MOVES_AWAY_START_MAX + 2
N_CUBES = int(1000 * ((2*SIDE*2)**(N_MOVES_AWAY_START_MAX - 1)))


def run(model_name):
    # Evaluation

    side = SIDE
    n_moves_low = 1
    n_moves_high = N_MOVES_AWAY_START_MAX

    n_moves_solve_max = N_MOVES_SOLVE_MAX

    total_count = np.zeros([n_moves_high - n_moves_low + 1])
    solved_count = np.zeros([n_moves_high - n_moves_low + 1])

    network_obj = network.load_network(model_name)

    total_states = 0
    seen_states = 0

    obs_list = []

    for i in range(N_CUBES):
        # NOTE: For testing more moves away cubes, just systematically generate rotations and moves from solved state or store cube states
        if n_moves_high > n_moves_low:
            n_moves = np.random.randint(n_moves_low, n_moves_high+1)
        else:
            n_moves = n_moves_low

        n_moves_solve_max = min(N_MOVES_SOLVE_MAX, n_moves + 2)
        # Create a fresh cube object with randomly initialized state
        C1 = cube.CubeObject(dim=side, n_moves=n_moves)

        obs_init = C1.get_observation()


        total_count[n_moves - n_moves_low] = total_count[n_moves - n_moves_low] + 1

        j = 0
        while (cube.isSolved(C1.state) is False) and (j < n_moves_solve_max):
            obs = C1.get_observation()
            #moves_encodings = np.zeros((1,1))
            #does_state_exist = True
            moves_encodings, does_state_exist = solver.network_step(obs, network_obj)

            total_states = total_states + 1
            if does_state_exist is True:
                seen_states = seen_states + 1

            moves = cube.decode_moves(moves_encodings, side)
            C1.apply_moves(moves)
            #cube.display(C1.state, C1.side, C1.colormap)
            j = j+1

        if cube.isSolved(C1.state) is True:
            solved_count[n_moves - n_moves_low] = solved_count[n_moves - n_moves_low] + 1
        else:
            obs_list += [obs_init]

    #print("Results:\nPercentage of seen states: "+str((100.0 * seen_states) / total_states))
    for i in range(n_moves_low, n_moves_high):
        percentage_solved = (100.0 * solved_count[i]) / total_count[i]
        print("Results:\nPercentage Solved "+str(i+n_moves_low+1)+" moves away: "+str(percentage_solved))

    percentage_solved = (100.0 * sum(solved_count)) / sum(total_count)
    print("Results:\nPercentage Solved overall: "+str(percentage_solved))
    return percentage_solved, obs_list


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "Q_Network_2STEP_20000_episodes.pickle"
    run(model_name)