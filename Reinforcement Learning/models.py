import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(in_features=250, out_features=3)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # nhwc -> nchw
        # center = x[:, :,:, :]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = x[:, :, 2:7, 2:7]
        x_center = center.flatten(start_dim=1)
        x_center = (self.fc1(x_center))
        return x_center

class PgMod(nn.Module):
    def __init__(self, num_actions):
        super(PgMod, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(90, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        x_center = center.flatten(start_dim=1)
        x_center = F.relu(self.linear1(x_center))
        x_center = F.relu(self.linear2(x_center))
        x_center = F.softmax(self.linear3(x_center), dim=1)
        return x_center
