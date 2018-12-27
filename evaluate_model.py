import numpy as np
import cube
import q_netowrk_solver
import q_network

n_moves_max = 2
side = 2

# Evaluation
n_moves_low = 1
n_moves_high = n_moves_max
model_name = "Q_Network_2STEP_20000_episodes.pickle"

total_count = np.zeros([n_moves_high - n_moves_low + 1])
solved_count = np.zeros([n_moves_high - n_moves_low + 1])

model = q_network.get_q_network_obj(model_name)


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
        obs = (C1.state).astype(np.int)
        moves_encodings, does_state_exist = q_netowrk_solver.q_network_step(obs, model)

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