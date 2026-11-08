import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, outs=64):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=outs),
            nn.ReLU(),
            nn.Linear(in_features=outs, out_features=outs),
            nn.ReLU(),
            nn.Linear(outs, action_dim),
        )

    def forward(self, x):
        return self.network(x)