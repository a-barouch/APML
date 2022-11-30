"""
A Snake game
A simple discrete control task. Run this file to see a random game.
The board is a 20x60 matrix. On the board there are 4 snakes (marked 1,2,3,4 - your is ‘1’),
    obstacles (marked ‘H’), and fruits (marked ‘*’, ‘$’ and 'X').
    Collision with obstacle or snake will kill your snake and it will restart itself in other place.

Each state is a 9x9 area around your snakes head, represented as one-hot vector. The state is aligned with the snake direction.

The possible actions are 0, 1, 2. o represent left, 1-forward, 2-right.

The reward is -5 for dieing and a different value for each fruit. In our version of the game the snake doesn't grow.

The game was originally created by Daniel Gissin, the previous TA. the game was changed a little this year.
"""

import gym
import time
import numpy as np
from snake import Game


# used for the interface with "snake_new" file, you don't need to use this.
int_to_action = {0: 'L', 1: 'F', 2: 'R'}
action_to_int = {'L': 0, 'F': 1, 'R': 2}


# this class has some functions of openai gym env, but doesn't follow the api.
class SnakeWrapper:
    """
    return the croped square_size-by-square_size after rotation and changing to one-hot and doing block-notation.
    """
    # num_classes is the number of different element types that can be found on the board.
    # yes I know, actually we have 9 types, but 10 is nicer. (4 snakes + 1 obstacle + 3 fruits + 1 empty = 9)
    num_classes = 10

    # the action space. 0-left, 1-forward, 2-right.
    action_space = gym.spaces.Discrete(3)

    # the observation space. 9x9 one hot vectors, total 9x9x10.
    # your snake always look up (the observation is a rotated crop of the board).
    observation_space = gym.spaces.Box(
        low=0,
        high=num_classes,
        shape=(9, 9, 10),
        dtype=np.int
    )

    def __init__(self):
        self.game = Game()
        self.square_size = 9 # the observation size
        self.timestep = 0

    def step(self, action):
        # get action as integer, move the game one step forward
        # return tuple: state, reward, done, info. done is always False - Snake game never ends.
        action = int_to_action[action]
        reward = self.game.step(action)

        head_pos = self.game.players[1].chain[-1]
        direction = self.game.players[1].direction
        board = self.game.board
        state = preprocess_snake_state(board, head_pos, direction, self.square_size, SnakeWrapper.num_classes)

        self.timestep += 1

        return state, reward

    def seed(self, seed=None):
        return self.game.seed(seed)

    # reset the game and return the board observation
    def reset(self):
        self.game.reset()
        self.timestep = 0
        first_state, _ = self.step(0)
        return first_state

    # print the board to the console
    def render(self, mode='human'):
        self.game.render(self.timestep)


def one_hot_2_d(x, num_classes):
    # get 2 dimensional array, return 3 dimensional one-hot. the last dimension has size num_classes
    # assume all values are in range(0, num_classes)
    shape = [x.shape[0], x.shape[1], num_classes]
    one_hot = np.zeros(shape, dtype=np.int)
    for h in range(x.shape[0]):
        for w in range(x.shape[1]):
            value = x[h, w]
            one_hot[h, w, value] = 1
    return one_hot


def preprocess_snake_state(board, head_pos, direction, square_size, num_classes):
    """
    return the croped square_size-by-square_size after rotation and changing to one-hot and doing block-notation.
    """
    # print(direction)
    # find the relevant square and rotate it
    rolled_board = np.roll(np.roll(board, -(head_pos[0]-int(square_size/2)), axis=0), -(head_pos[1]-int(square_size/2)))
    square = rolled_board[0:square_size, 0:square_size]
    orig_square = square
    if direction == 'UP':
        pass
    elif direction == 'LEFT':
        square = np.rot90(square, k=3, axes=(0, 1))
    elif direction == 'DOWN':
        square = np.rot90(square, k=2, axes=(0, 1))
    elif direction == 'RIGHT':
        square = np.rot90(square, k=1, axes=(0, 1))
    else:
        raise Exception

    data = one_hot_2_d(square, num_classes=10)
    # return data
    return np.expand_dims(data, 0)


if __name__ == '__main__':
    game = SnakeWrapper()
    action_space = SnakeWrapper.action_space
    for i in range(100):
        action = action_space.sample()
        print(f'the {i} action is {action}')
        state, reward = game.step(action)
        print('rendered board:')
        game.render()
        time.sleep(0.5)



