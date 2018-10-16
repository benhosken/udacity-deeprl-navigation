import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        
        
        # A range of model sizes have been tested from 2 layers to 4 with sizes varying from 32 to 128 across the 
        # tests. Overall, there was negliable difference between the various models. 
        # We have chosen to use a simple model with 2 hidden layers
        
        fc1_units=64
        fc2_units=128
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        #self.fc3 = nn.Linear(fc2_units, fc3_units)
        #self.fc3.weight.data.normal_(0, 0.1)   # initialization
        #self.fc4 = nn.Linear(fc3_units, fc4_units)
        #self.fc4.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(fc2_units, action_size)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        return self.out(x)
