import torch
import torch.optim as optim
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from base_policy import BasePolicy
import gym
from models import SimpleModel


class QPolicy(BasePolicy):
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

        self.target_net = SimpleModel()
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        random_number = random.random()
        if random_number > epsilon:
            with torch.no_grad():
                return self.model.forward(state).max(1)[1][0].item(), None

        else:
            return self.action_space.sample(), None  # return action randomly

    # partly relied on the following tutorial:
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    def optimize(self, batch_size, global_step=None):
        # optimize your model

        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:

            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.model(state_batch)
            state_action_values = state_action_values.gather(dim=1, index=action_batch.unsqueeze(1))
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute loss todo maybe change loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # normalize the gradient
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.target_net.load_state_dict(self.model.state_dict())
