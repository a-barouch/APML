import shutil
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import argparse
import os
import sys
import matplotlib.pyplot as plt

from q_policy import QPolicy
from pg_policy import MCPolicy
from snake_wrapper import SnakeWrapper
from models import SimpleModel, PgMod


def create_model(network_name):
    """
    Kind-of model factory.
    Edit it to add more models.
    :param network_name: The string input from the terminal
    :return: The model
    """
    if network_name == 'simple':
        return SimpleModel()
    if network_name == 'pg_mod':
        return PgMod(num_actions=3)
    else:
        raise Exception('net {} is not known'.format(network_name))


def create_policy(policy_name,
                  buffer_size, gamma, model: torch.nn.Module, writer: SummaryWriter, lr):
    """
    Kind-of policy factory.
    Edit it to add more algorithms.
    :param policy_name: The string input from the terminal
    :param buffer_size: size of policy's buffer
    :param gamma: reward decay factor
    :param model: the pytorch model
    :param writer: tensorboard summary writer
    :param lr: initial learning rate
    :return: A policy object
    """
    if policy_name == 'dqn':
        return QPolicy(buffer_size, gamma, model, SnakeWrapper.action_space, writer, lr=lr)
    if policy_name == 'pg':
        return MCPolicy(buffer_size, gamma, model, SnakeWrapper.action_space, writer, lr=lr)
    else:
        raise Exception('algo {} is not known'.format(policy_name))


def train(steps, buffer_size, opt_every,
          batch_size, lr, max_epsilon, policy_name, gamma, network_name,
          log_dir):

    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    policy = create_policy(policy_name, buffer_size, gamma, model, writer, lr)

    state = game.reset()
    state_tensor = torch.FloatTensor(state)
    reward_history = []
    prob_history = []
    ep_rewards = []
    mean_rewards = []

    for step in tqdm(range(steps)):

        # epsilon exponential decay
        epsilon = max_epsilon * math.exp(-1. * step / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, step)

        prev_state_tensor = state_tensor
        action, prob = policy.select_action(state_tensor, epsilon)
        state, reward = game.step(action)

        reward_history.append(reward)
        prob_history.append(prob)
        ep_rewards.append(reward)

        state_tensor = torch.FloatTensor(state)
        reward_tensor = torch.FloatTensor([reward])
        action_tensor = torch.LongTensor([action])

        policy.record(prev_state_tensor, action_tensor, state_tensor, reward_tensor)

        writer.add_scalar('training/reward', reward_history[-1], step)

        if step % opt_every == opt_every - 1:
            print("mean reward: " + str(sum(ep_rewards)/len(ep_rewards)))
            mean_rewards.append(sum(ep_rewards)/len(ep_rewards))
            if policy_name=="dqn":
                send_to_optimize = batch_size
            elif policy_name == "pg":
                send_to_optimize = (ep_rewards,prob_history)
            policy.optimize(send_to_optimize)
            ep_rewards = []
            prob_history = []
    writer.close()


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--name', type=str, required=True, help='the name of this run')
    p.add_argument('--log_dir', type=str, required=True, help='directory for tensorboard logs (common to many runs)')

    # loop
    p.add_argument('--steps', type=int, default=10000, help='steps to train')
    p.add_argument('--opt_every', type=int, default=100, help='optimize every X steps')

    # opt
    p.add_argument('--buffer_size', type=int, default=800)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('-e', '--max_epsilon', type=float, default=0.1, help='for pg, use max_epsilon=0')
    p.add_argument('-g', '--gamma', type=float, default=.1)
    p.add_argument('-p', '--policy_name', type=str, choices=['dqn', 'pg', 'a2c'], required=True)
    p.add_argument('-n', '--network_name', type=str, choices=['simple', 'pg_mod'], required=True)

    args = p.parse_args()
    return args


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.name)

    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
        # if query_yes_no('You already have a run called {}, override?'.format(args.name)):
        #     shutil.rmtree(args.log_dir)
        # else:
        #     exit(0)

    del args.__dict__['name']
    train(**args.__dict__)


