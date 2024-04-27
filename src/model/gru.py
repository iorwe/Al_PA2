import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__ (self,hidden_dim,num_layers=1,bidirectional=False,embedding_dim=50,num_classes=2,dropout=0.5):
        super(GRU, self).__init__()

        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        # GRU layer
        x, _ = self.gru(x)
        if self.gru.bidirectional:
            # Concatenate the final forward and backward hidden state
            x = torch.cat((x[:,-1,:self.gru.hidden_size], x[:,0,self.gru.hidden_size:]), dim=-1)
        else:
            x = x[:,-1]
        # Dropout
        x = self.dropout(x)
        # Fully connected layer to get class logits
        logits = self.fc(x)

        return logits