import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,hidden_dim,num_layers=1,bidirectional=False,embedding_dim=50,num_classes=2,dropout=0.5):
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        # LSTM layer
        x, _ = self.lstm(x)
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden state
            x = torch.cat((x[:,-1,:self.lstm.hidden_size], x[:,0,self.lstm.hidden_size:]), dim=-1)
        else:
            x = x[:,-1]
        # Dropout
        x = self.dropout(x)
        # Fully connected layer to get class logits
        logits = self.fc(x)

        return logits