import torch.nn as nn
import torch.optim as optim


class DoubleDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DoubleDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        return self.layers(x)