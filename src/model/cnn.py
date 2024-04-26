import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self,num_filters,filter_sizes,embedding_dim=50,num_classes=2,dropout=0.5):
        super(CNN, self).__init__()

        # Convolutional layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes]
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # Add a channel dimension
        x = x.unsqueeze(1)
        # Apply convolution and ReLU activation
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # Apply max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # Concatenate the filtered tensors
        x = torch.cat(x, 1)
        # Dropout
        x = self.dropout(x)
        # Fully connected layer to get class logits
        logits = self.fc(x)
        return logits
    
# # Example parameters
# num_filters = 100  # Number of filters per filter size
# filter_sizes = [2,3,4]  # Different filter sizes
# # Create a model instance
# model = CNN(num_filters, filter_sizes)
# print(model)
