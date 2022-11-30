# based on "Snake" game used in APML in previous years. Thanks to Daniel Gissin for most of the code.

import numpy as np
import argparse
import os
import scipy.signal as ss  # we are using convolution to find empty places on the board
from gym.utils import seeding

# for Demo
from pynput.keyboard import Key, Listener
import time


# board values: [0, 8] inclusive
EMPTY_VAL = 0
SNAKES_VALUES = [1, 2, 3, 4] # you are always number one!
OBSTACLE_VAL = 5
REGULAR_RENDER_MAP = {EMPTY_VAL: ' ', OBSTACLE_VAL: 'H'}
FOOD_RENDER_MAP = {6:'*', 7:'$', 8:'X'}
FOOD_VALUE_MAP = {6:1, 7:3, 8:0}
FOOD_REWARD_MAP = {6:2, 7:5, 8:-1}
THE_DEATH_PENALTY = -5

ILLEGAL_MOVE = "Illegal Action: the default action was selected instead. Player tried action: "
NO_RESPONSE = "No Response: player took too long to respond with action. This is No Response #"
UNRESPONSIVE_PLAYER = "Unresponsive Player: the player hasn't responded in too long... SOMETHING IS WRONG!!"


DEFAULT_ACTION = 'F'

ACTIONS = ['L',  # counter clockwise (left)
           'R',  # clockwise (right)
           'F']  # forward
TURNS = {
    # if we start in direction 'key_0', action 'key_1' will bring us to direction 'value'.
    'UP':    {'L': 'LEFT',  'R': 'RIGHT', 'F': 'UP'},
    'DOWN':  {'L': 'RIGHT', 'R': 'LEFT',  'F': 'DOWN'},
    'LEFT':  {'L': 'DOWN',  'R': 'UP',    'F': 'LEFT'},
    'RIGHT': {'L': 'UP',    'R': 'DOWN',  'F': 'RIGHT'}
}


class Position:
    def __init__(self, position, board_size):
        self.pos = position
        self.board_size = board_size

    def __getitem__(self, key):
        return self.pos[key]

    def __repr__(self):
        return '({},{})'.format(self.pos[0], self.pos[1])

    def __add__(self, other):
        return Position(((self[0] + other[0]) % self.board_size[0],
                        (self[1] + other[1]) % self.board_size[1]),
                        self.board_size)

    def move(self, dir):
        if dir == 'RIGHT': return self + (0,1) #E
        if dir == 'LEFT' : return self + (0,-1) #w
        if dir == 'UP'   : return self + (-1, 0) #n
        if dir == 'DOWN' : return self + (1, 0) #s
        raise ValueError('unrecognized direction')


class Player:
    """
    the game hold players.
    each player knows his locations on the board, comulative reward, etc
    """
    initial_size = 3
    initial_growth = 2

    def __init__(self):
        self.chain = []  # each position is a tuple
        self.reward_history = []
        self.growth = Player.initial_growth
        self.direction = None
        # size is just len(positions), no need for variable

    def reset(self, chain, direction):
        self.chain = chain
        self.growth = Player.initial_growth
        self.direction = direction
        # reward stays!!

    def chain_as_np_array(self):
        chain_as_list_of_tuples = [(e[0], e[1]) for e in self.chain]
        array = np.array(chain_as_list_of_tuples)
        return array

    def move_snake(self, pid, action):
        # delete the tail if the snake isn't growing:
        if self.growth > 0:
            self.growth -= 1
        else:
            del self.chain[0]

        # move the head:
        self.direction = TURNS[self.direction][action]
        new_position = self.chain[-1].move(self.direction)
        self.chain.append(new_position)

        return new_position


class Game:
    """
    main game class.
    you are playing against 3 opponent agents,
     that always plays "Avoid" policies with 0.5, 0.1, 0.0
    """
    actions = ['L', 'F', 'R']

    def __init__(self):
        self.args = self.parse_args()
        self.board = None
        self.players = None
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def parse_args(self):
        # this method was edited such that all the arguments are constant.
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        args = AttrDict()
        args['board_h'] = 20
        args['board_w'] = 60
        args['obstacle_density'] = 0.1
        args['max_item_density'] = 0.05

        # p.add_argument('--board_h', '-bh', type=int, default=20, help='a tuple of (height, width)')
        # p.add_argument('--board_w', '-bw', type=int, default=60, help='a tuple of (height, width)')
        # p.add_argument('--obstacle_density', '-od', type=float, default=.1,
        #                help='the density of obstacles on the board')
        # p.add_argument('--max_item_density', '-mid', type=float, default=.05,
        #                help='maximum item density in the board (not including the players)')

        args['board_size'] = (args['board_h'], args['board_w'])
        args.__dict__ = args
        return args

    def find_empty_slot(self, shape):
        # find an empty slot and return it's position.
        empty_map = self.board == EMPTY_VAL
        match = ss.convolve2d(empty_map, np.ones(shape), mode='same', boundary='wrap') == np.prod(shape)
        if not np.any(match):
            return None

        good_positions = np.argwhere(match)  # shape is (N, 2) where N is the number of good positions
        idx = self.np_random.randint(good_positions.shape[0])
        position = good_positions[idx, :]
        position = (position[0], position[1])
        position = Position(position, self.args.board_size)

        return position

    def create_board(self):
        # create board
        self.board = np.zeros(shape=(self.args.board_h, self.args.board_w), dtype=np.int)
        # put obstacles
        while np.sum(self.board == OBSTACLE_VAL) < self.args.board_h * self.args.board_w * self.args.obstacle_density:
            # randomly choose part size and orientation
            small_edge = min(self.args.board_h, self.args.board_w)
            size = self.np_random.choice(np.arange(small_edge // 4, small_edge // 2))
            orientation = self.np_random.choice(['horizontal', 'vertical'])
            if orientation == 'horizontal':
                obstacle_shape = (1, size)
            else:
                obstacle_shape = (size, 1)

            position = self.find_empty_slot(obstacle_shape)
            if orientation == 'horizontal':
                for i in range(size):
                    self.board[position[0], (position[1] + i) % self.args.board_w] = OBSTACLE_VAL
            else:
                for i in range(size):
                    self.board[(position[0] + i) % self.args.board_h, position[1]] = OBSTACLE_VAL

    def remove_snake_from_board(self, pid):
        self.board[self.board == pid] = EMPTY_VAL

    def put_snake_on_board(self, pid):
        self.board[self.players[pid].chain_as_np_array()[:, 0], self.players[pid].chain_as_np_array()[:, 1]] = pid

    def restart_player(self, pid):  # every time a snake dies

        # remove snake from board
        if self.players[pid].chain:
            self.remove_snake_from_board(pid)

        assert not np.any(self.board == pid)

        # compost the snake into fruits - not implemented!
        pass

        # randomize position and direction
        direction = self.np_random.choice(list(TURNS.keys()))
        shape = (1, Player.initial_size) if direction in ['RIGHT', 'LEFT'] else (3, 1)
        pos = self.find_empty_slot(shape)

        # create chain
        chain = [pos]
        for i in range(Player.initial_size - 1):
            pos = pos.move(direction)
            chain.append(pos)

        self.players[pid].reset(chain, direction)

        # put player on the board
        self.put_snake_on_board(pid)

    def reset(self):  # create a whole new board and players (with zero reward!)
        self.create_board()
        self.players = {}
        for i in range(1, 5):
            self.players[i] = Player()  # number one is the user player
            self.restart_player(i)

        enough_food = False
        while not enough_food:
            enough_food = not self.add_food_randomly()

    def avoid_action(self, pid, epsilon):

        head_position = self.players[pid].chain[-1]
        head_direction = self.players[pid].direction

        if self.np_random.rand() < epsilon:
            return self.np_random.choice(ACTIONS)

        else:
            for a in list(self.np_random.permutation(ACTIONS)):

                # get a Position object of the position in the relevant direction from the head:
                next_position = head_position.move(TURNS[head_direction][a])
                r = next_position[0]
                c = next_position[1]

                # look at the board in the relevant position:
                if self.board[r, c] not in SNAKES_VALUES + [OBSTACLE_VAL]:
                    return a

            # if all positions are bad:
            return self.np_random.choice(ACTIONS)

    def step_single_snake(self, pid, action):

        pos = self.players[pid].move_snake(pid, action)
        prev_board_value = self.board[pos[0], pos[1]]

        if prev_board_value in SNAKES_VALUES + [OBSTACLE_VAL]:  # snakes died
            self.players[pid].reward_history.append(THE_DEATH_PENALTY)
            self.restart_player(pid)
        elif prev_board_value in FOOD_REWARD_MAP.keys():        # snake eat
            self.players[pid].reward_history.append(FOOD_REWARD_MAP[prev_board_value])
        else:                                                   # snake moves to an empty place
            self.players[pid].reward_history.append(0)

        self.remove_snake_from_board(pid)
        self.put_snake_on_board(pid)

    def add_food_randomly(self):
        item_count = {}
        total_item_count = 0
        for key in FOOD_VALUE_MAP.keys():
            count = np.sum(self.board == key)
            item_count[key] = count
            total_item_count += count

        if total_item_count < self.args.max_item_density * np.prod(self.args.board_size):
            # if we choose the food randomly, somewhen we will have too much "bad foods".
            # so, we add food from the type with the minimal count.
            # randfood = self.np_random.choice(list(FOOD_VALUE_MAP.keys()), 1) # no longer in use!
            randfood = min(item_count, key=item_count.get)
            slot = self.find_empty_slot((1, 1))
            if not slot:
                raise Exception('no empty place on the board, should not happen...')
            self.board[slot[0], slot[1]] = randfood
            return True

        else:  # already too much food
            return False

    def step(self, action):
        # every player choose action and act, the input action is for player 1
        if not action:
            action = self.np_random.choice(ACTIONS)

        self.step_single_snake(1, action)

        epsilons = [0.5, 0.1, 0.0]
        for i, epsilon in enumerate(epsilons):
            pid = i + 2
            action = self.avoid_action(pid, epsilon)
            self.step_single_snake(pid, action)

        self.add_food_randomly()
        return self.players[1].reward_history[-1]

    @property
    def render_map(self):
        # configure symbols for rendering
        render_map = {pid: str(pid) for pid in self.players.keys()}
        render_map.update(REGULAR_RENDER_MAP)
        render_map.update(FOOD_RENDER_MAP)
        return render_map

    def render_board(self, board_np, timestep):
        if os.name == 'nt':
            os.system('cls') # clear screen for Windows
        else:
            print(chr(27)+"[2J") # clear screen for linux

        # print the scores:
        print('Time Step: {}'.format(str(timestep)))
        score_scope = min(timestep, 100)
        for pid in self.players.keys():
            score = np.mean(self.players[pid].reward_history[-score_scope:])
            print('Player {}: {}'.format(pid, score))
        print('todo: verify')

        # print the board:
        horzline = '-' * (board_np.shape[1] + 2)
        board = [horzline]
        for timestep in range(board_np.shape[0]):
            # board.append('|' +''.join(self.render_map[self.board[timestep, c]] for c in range(self.board.shape[1])) + '|')
            board.append('|' +''.join(self.render_map[board_np[timestep, c]] for c in range(board_np.shape[1])) + '|')
        board.append(horzline)
        print('\n'.join(board))

    def render(self, timestep):
        if os.name == 'nt':
            os.system('cls') # clear screen for Windows
        else:
            print(chr(27)+"[2J") # clear screen for linux

        # print the scores:
        print('Time Step: {}'.format(str(timestep)))
        score_scope = min(timestep, 100)
        for pid in self.players.keys():
            score = np.mean(self.players[pid].reward_history[-score_scope:])
            print('Player {}: {}'.format(pid, score))
        print('todo: verify')

        # print the board:
        horzline = '-' * (self.board.shape[1] + 2)
        board = [horzline]
        for timestep in range(self.board.shape[0]):
            # board.append('|' +''.join(self.render_map[self.board[timestep, c]] for c in range(self.board.shape[1])) + '|')
            board.append('|' +''.join(self.render_map[self.board[timestep, c]] for c in range(self.board.shape[1])) + '|')
        board.append(horzline)
        print('\n'.join(board))


if __name__ == '__main__':

    # configure interface

    pressed_key = None
    key_to_action = {Key.left: 'L', Key.right: 'R'}

    def on_press(key):
        global pressed_key
        if key in key_to_action.keys():
            pressed_key = key

    def on_release(key):
        global pressed_key
        if key == pressed_key:
            pressed_key = None

    # Collect events until released
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        # listener.join()

        game = Game()

        for i in range(1000):

            game.render(0)

            if pressed_key is None:
                action = 'F'
            else:
                action = key_to_action[pressed_key]
            game.step(action)
            time.sleep(0.5)

        # listener.join()
