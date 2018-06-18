import numpy as np
from matplotlib import pyplot as plt

FRONT = 1
LEFT = 0
RIGHT = 2
BACK = 3
TOP = 4
BOTTOM = 5

class cube:
    def __init__(self, dim, n_moves = 100):
        self.side = dim
        self.area = self.side * self.side
        self.cube = np.zeros([6,self.side,self.side]).astype(np.int) # Order: Left,Front,Right,Back,Up,Down - [0,1,2,3,4,5] - [Orange, Green, Red, Blue, White, Yellow]
        self.colormap = np.array([[255,165,000],[000,128,000],[255,000,000],[255,255,000],[255,255,255],[000,000,255]]).astype(np.uint8)
        self.moves = np.zeros([0, 3]).astype(np.int)
        self.cube_states = np.zeros([0,6,self.side,self.side])

        for i in range(6):
            # Initialize a solved cube
            # Top face is 0, side faces are 1-4
            # Bottom face is 5
            self.cube[i,:,:] = i

        # self.display()
        self.random_shuffle(n_moves)
        # self.display()

    def display(self):
        # Change according to convention
        x = np.zeros([self.side * 3, self.side * 4, 3]).astype(np.uint8)
        x[:,:,:] = 200

        for i in range(6):
            if i < 4:
                # Sides
                startx = i * self.side
                starty = self.side
                for j in range(starty, starty + self.side):
                    for k in range(startx, startx + self.side):
                        x[j,k,:] = self.colormap[self.cube[i,j - starty,k - startx]]

            if i == 4:
                # Top
                startx = self.side
                starty = 0
                for j in range(starty, starty + self.side):
                    for k in range(startx, startx + self.side):
                        x[j,k,:] = self.colormap[self.cube[i,j - starty,k - startx]]

            if i == 5:
                # Bottom
                startx = self.side
                starty = 2 * self.side
                for j in range(starty, starty + self.side):
                    for k in range(startx, startx + self.side):
                        x[j,k,:] = self.colormap[self.cube[i,j - starty,k - startx]]

        plt.imshow(x)
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, 4 * self.side, 1))
        ax.set_yticks(np.arange(-.5, 3 * self.side, 1))
        ax.grid(color='k', linestyle='-', linewidth=2)

    def get_inverse_moves(self, moves):
        n_moves = moves.shape[0]
        temp = moves[::-1,:].copy()
        for i in range(n_moves):
            temp[i, 0:2] = moves[n_moves - 1 - i, 0:2]
            temp[i, 2] = 1 - moves[n_moves - 1 - i, 2]

        return temp

    def moves_shuffle(self, moves):
        n_moves = moves.shape[0]
        for i in range(n_moves):
            self.move(axis=moves[i,0].astype(np.int), offset=moves[i,1].astype(np.int), direction=moves[i,2].astype(np.int))
        return

    def random_shuffle(self, n_moves):
        moves = np.zeros([n_moves,3])
        for i in range(n_moves):
            axis = np.random.randint(0,3)
            offset = np.random.randint(0,self.side)
            direction = np.random.randint(0,2)
            moves[i,:] = [axis, offset, direction]
        self.moves_shuffle(moves)
        return

    def rotate_turn_counterclockwise(self, axis):
        temp = np.zeros(self.cube.shape).astype(np.int)
        if axis == 1:
            # Y-Axis
            temp[LEFT,:,:] = self.cube[BACK,:,:]
            temp[FRONT,:,:] = self.cube[LEFT,:,:]
            temp[RIGHT,:,:] = self.cube[FRONT,:,:]
            temp[BACK,:,:] = self.cube[RIGHT,:,:]

            temp[TOP,:,:] = np.rot90(self.cube[TOP,:,:], 1)
            temp[BOTTOM, :, :] = np.rot90(self.cube[BOTTOM, :, :], 3)
        elif axis == 0:
            # X-Axis
            temp[BACK, :, :] = self.cube[TOP, ::-1, ::-1]
            temp[TOP, :, :] = self.cube[FRONT, :, :]
            temp[FRONT, :, :] = self.cube[BOTTOM, :, :]
            temp[BOTTOM, :, :] = self.cube[BACK, ::-1, ::-1]

            temp[LEFT, :, :] = np.rot90(self.cube[LEFT, :, :], 1)
            temp[RIGHT, :, :] = np.rot90(self.cube[RIGHT, :, :], 3)
        else:
            # Z-Axis
            temp[LEFT, :, :] = self.cube[TOP, :, :]
            temp[TOP, :, :] = self.cube[RIGHT, :, :]
            temp[RIGHT, :, :] = (self.cube[BOTTOM, :, :].transpose())[::-1,::-1]
            temp[BOTTOM, :, ::-1] = (self.cube[LEFT, :, :].transpose())[::-1,:]

            temp[FRONT, :, :] = np.rot90(self.cube[FRONT, :, :], 1)
            temp[BACK, :, :] = np.rot90(self.cube[BACK, :, :], 3)

        self.cube = temp
        return

    def rotate_cube(self, axis, turns):
        TURNS = np.arange(0,4)
        turns = TURNS[np.remainder(turns, 4)]
        for i in range(turns):
            self.rotate_turn_counterclockwise(axis)
        return

    def moveY(self, offset, direction):
        # DO NOT CALL THIS FUNCTION FROM ANYWHERE BUT THE move() ROUTINE!!!!!
        # Does only a 90 degree rotation along Y axis only
        # Direction - 0 (Clockwise when looking along that axes), 1 (Anti-Clockwise)
        # Standard Right hand axes convention. X (-->) Y (V - downward) Z (x - Into the plane).
        # For any face, (0,0) is always the top left according to the figure in:
        # https://ruwix.com/the-rubiks-cube/japanese-western-color-schemes/
        # Offset is the offset from the face that comes first along that axis - ex: offset 0 for Y Axis is TOP

        # Faces
        if offset == 0 or offset == self.side - 1:
            # Along with sides, even face needs rotation
            if offset == 0:
                face = TOP
                f = self.cube[face,:,:]
                if direction == 1:
                    f1 = np.rot90(f, 1)  # 90 degrees anti-clockwise
                else:
                    f1 = np.rot90(f, 3)  # 270 degrees anti-clockwise
            else:
                face = BOTTOM
                f = self.cube[face,:,:]
                if direction == 1:
                    f1 = np.rot90(f, 3)  # 270 degrees anti-clockwise
                else:
                    f1 = np.rot90(f, 1)  #  90 degrees anti-clockwise
            self.cube[face,:,:] = f1

        # Sides
        a = self.cube[FRONT, offset, :].copy()
        b = self.cube[RIGHT, offset, :].copy()
        c = self.cube[BACK,  offset, :].copy()
        d = self.cube[LEFT,  offset, :].copy()

        if direction == 1:
            # Anti-clockwise
            self.cube[FRONT, offset, :] = d
            self.cube[RIGHT, offset, :] = a
            self.cube[BACK,  offset, :] = b
            self.cube[LEFT,  offset, :] = c
        else:
            # Positive
            self.cube[FRONT, offset, :] = b
            self.cube[RIGHT, offset, :] = c
            self.cube[BACK,  offset, :] = d
            self.cube[LEFT,  offset, :] = a

        return

    def move(self, axis, offset, direction):
        # Does only a 90 degree rotation
        # Direction - 0 (Clockwise when looking along that axes), 1 (Anti-Clockwise)
        # Standard Right hand axes convention. X (-->) Y (V - downward) Z (x - Into the plane).
        # For any face, (0,0) is always the top left according to the figure in:
        # https://ruwix.com/the-rubiks-cube/japanese-western-color-schemes/
        # Offset is the offset from the face that comes first along that axis - ex: offset 0 for Y Axis is TOP

        # Use Y axis rotation and cube rotations as basis
        if axis == 1:
            # Y-Axis
            self.moveY(offset, direction)
        if axis == 0:
            # X-Axis
            self.rotate_cube(1, 1)
            self.rotate_cube(0, 1)
            self.moveY(offset, direction)
            self.rotate_cube(0,-1)
            self.rotate_cube(1,-1)
        if axis == 2:
            # Z-Axis
            self.rotate_cube(0,1)
            self.moveY(offset, direction)
            self.rotate_cube(0,-1)

        self.moves = np.append(self.moves, [np.asarray([axis,offset,direction])], axis=0)
        self.cube_states = np.append(self.cube_states, [self.cube], axis=0)
        return



    def isSolved(self):
        # returns whether the cube is solved
        return

    def orient(self):
        # Rotate the entire cube in a way that a fixed orientation is preserved
        return