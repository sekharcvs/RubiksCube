import cube
import numpy as np
from keras.models import load_model

# Set these parameters
side = 2
n_moves = 1
# Order: Left,Front,Right,Back,Up,Down - [0,1,2,3,4,5] - [Orange (0), Green (1), Red (2), Yellow (3), White (4), Blue (5)]
initial_state = np.asarray([
    [[0,5], [3,3]], # Left
    [[2,5], [1,0]], # Front
    [[4,0], [4,2]], # Right
    [[4,3], [1,1]], # Back
    [[5,5], [3,2]], # Top
    [[2,1], [0,4]], # Bottom
])


model_name = "model_5moves.h5"
model = load_model(model_name)

C1 = cube.CubeObject(dim=side, n_moves=n_moves)
C1.set_state(initial_state)
#cube.display(C1.state, C1.side, C1.colormap)
j = 0
moves_list = []
while (cube.isSolved(C1.state) is False) and (j < 21):
    cube_state = np.asarray([C1.state]).astype(np.int)
    moves_encodings = model.predict_classes(cube_state)
    moves = cube.decode_moves(moves_encodings, side)
    moves_list = moves_list + [moves]
    C1.apply_moves(moves)
    #cube.display(C1.state, C1.side, C1.colormap)
    j = j + 1

print(moves_list)
