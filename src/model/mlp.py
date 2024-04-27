import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim=2, input_dim=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*input_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x