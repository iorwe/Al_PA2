import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__ (self,hidden_dim,num_layers=1,bidirectional=False,embedding_dim=50,num_classes=2,dropout=0.5):
        super(RNN, self).__init__()

        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        # RNN layer
        x, _ = self.rnn(x)
        if self.rnn.bidirectional:
            # Concatenate the final forward and backward hidden state
            x = torch.cat((x[:,-1,:self.rnn.hidden_size], x[:,0,self.rnn.hidden_size:]), dim=-1)
        else:
            x = x[:,-1]
        # Dropout
        x = self.dropout(x)
        # Fully connected layer to get class logits
        logits = self.fc(x)

        return logits