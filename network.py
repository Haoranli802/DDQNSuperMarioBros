import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """
    The most primitive Q-learning algorithms always require a Q-table for recording during execution, which can be sufficient when the number of dimensions is low.
        However, when exponential dimensionality is encountered, the efficiency of the Q-table becomes very limited. Therefore, we consider a value function approximation approach that achieves
        The corresponding Q-value can be obtained in real time each time by simply knowing S or A in advance.

        Originally, the Q value was found by S (state), A (action), but now it is implemented using a neural network to obtain an approximate Q value.
        Neural networks are very powerful, so it is possible to take the game screen as a state S and feed it directly into the network, and we can make the output of the network the Q value
        So that each output corresponds to the Q value of an action A, we can obtain the Q value of each action in state S. This is the DQN

        For this Mario game, if we were to look at a single frame alone we would be missing some information, for example we could not tell from a single frame whether Mario
        is going up or down. So it is estimated that if several consecutive frames are input at the same time, the neural network will be able to tell whether Mario is going up or down, and it can be estimated that
        that this would work better as an input. This is the reason why the previous classes that inherit from gym have been adapted to Mario's output game screen.

    """
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)