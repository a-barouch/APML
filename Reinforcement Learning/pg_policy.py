import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
import numpy as np


class MCPolicy(BasePolicy):
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(MCPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        probs = self.model.forward(state)
        self.epsilon = epsilon
        # choose an action randomly by probability of each action
        highest_prob_action = np.random.choice(self.model.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def optimize(self, args, global_step=None):
        discounted_rewards = []
        rewards, log_probs = args

        # calculate Vt's
        for t in range(len(rewards)):
            Vt = 0
            pw = 0
            for r in rewards[t:]:
                Vt = Vt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Vt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)         # normalize discounted rewards

        # calculate entropy
        entropy = []
        for p in log_probs:
            p = np.exp(p.item()) # from log prob to prob
            entropy.append(np.log(1 / p) * p)

        # calculate objective to maximize
        objective = []
        for log_prob, Vt, ent in zip(log_probs, discounted_rewards, entropy):
            objective.append(-log_prob * Vt + ent * self.epsilon)

        # maximize objective and update weights accordingly
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        objective = torch.stack(objective).sum()
        objective.backward()
        self.optimizer.step()
