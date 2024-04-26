import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes]
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # Embedding input words
        x = self.embedding(x)  # [batch_size, sentence_length, embedding_dim]

        # Add a channel dimension
        x = x.unsqueeze(1)  # [batch_size, 1, sentence_length, embedding_dim]

        # Apply convolution and ReLU activation
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # List of [batch_size, num_filters, convolved_length]

        # Apply max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # List of [batch_size, num_filters]

        # Concatenate the filtered tensors
        x = torch.cat(x, 1)  # [batch_size, num_filters * num_of_filters]

        # Dropout
        x = self.dropout(x)

        # Fully connected layer to get class logits
        logits = self.fc(x)  # [batch_size, num_classes]

        return logits

# Example parameters
vocab_size = 10000  # Size of vocabulary
embedding_dim = 300  # Dimension of word embeddings
num_filters = 100  # Number of filters per filter size
filter_sizes = [3, 4, 5]  # Different filter sizes
num_classes = 2  # Number of classes (e.g., positive and negative)

# Create a model instance
model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes)
print(model)
