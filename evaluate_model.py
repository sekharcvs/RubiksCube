import numpy as np
import cube
import dqn_network_solver as solver
import dqn_network as network

n_moves_max = 2
side = 2


def run(model_name):
    # Evaluation
    n_moves_low = 1
    n_moves_high = n_moves_max

    total_count = np.zeros([n_moves_high - n_moves_low + 1])
    solved_count = np.zeros([n_moves_high - n_moves_low + 1])

    network_obj = network.load_network(model_name)

    total_states = 0
    seen_states = 0
    for i in range(1000):
        if n_moves_high > n_moves_low:
            n_moves = np.random.randint(n_moves_low, n_moves_high+1)
        else:
            n_moves = n_moves_low

        C1 = cube.CubeObject(dim=side, n_moves=n_moves)
        #cube.display(C1.state, C1.side, C1.colormap)

        total_count[n_moves - n_moves_low] = total_count[n_moves - n_moves_low] + 1

        j = 0
        while (cube.isSolved(C1.state) is False) and (j <3):
            obs = C1.get_observation()
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

    print("Results:\nPercentage of seen states: "+str((100.0 * seen_states) / total_states))
    for i in range(0, n_moves_high):
        percentage_solved = (100.0 * solved_count[i]) / total_count[i]
        print("Results:\nPercentage Solved "+str(i+1)+" moves away: "+str(percentage_solved))

    percentage_solved = (100.0 * sum(solved_count)) / sum(total_count)
    print("Results:\nPercentage Solved overall: "+str(percentage_solved))


if __name__ == "__main__":
    run("Q_Network_2STEP_20000_episodes.pickle")