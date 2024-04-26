import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model.cnn import CNN
from src.dataset import TextDataset

def train(model, train_loader, optimizer, criterion, device):
    # Set the model to training mode
    model.train()
    # Initialize the running loss
    train_loss = 0.
    train_tp, train_fp, train_tn, train_fn = 0.0, 0.0, 0.0, 0.0
    total_samples = 0.

    # Iterate over the training data
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # Update the running loss
        train_loss += loss.item()*inputs.size(0)
        # Calculate the accuracy
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        train_tp += (predicted * targets).sum().item()
        train_fp += (predicted * (1 - targets)).sum().item()
        train_tn += ((1 - predicted) * (1 - targets)).sum().item()
        train_fn += ((1 - predicted) * targets).sum().item()
    
    # Calculate the average loss and accuracy
    train_accuracy = (train_tp + train_tn) / total_samples
    train_precision = train_tp / (train_tp + train_fp)
    train_recall = train_tp / (train_tp + train_fn)
    train_loss = train_loss / total_samples
    F_score = 2.0 * train_precision * train_recall / (train_precision + train_recall)
    return train_loss, train_accuracy, F_score
        
def validate(model, valid_loader, criterion, device):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the running loss
    valid_loss = 0.
    valid_tp, valid_fp, valid_tn, valid_fn = 0.0, 0.0, 0.0, 0.0
    total_samples = 0.

    with torch.no_grad():
        # Iterate over the validation data
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Update the running loss
            valid_loss += loss.item()*inputs.size(0)
            # Calculate the accuracy
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            valid_tp += (predicted * targets).sum().item()
            valid_fp += (predicted * (1 - targets)).sum().item()
            valid_tn += ((1 - predicted) * (1 - targets)).sum().item()
            valid_fn += ((1 - predicted) * targets).sum().item()

    # Calculate the average loss and accuracy
    valid_accuracy = (valid_tp + valid_tn) / total_samples
    valid_precision = valid_tp / (valid_tp + valid_fp)
    valid_recall = valid_tp / (valid_tp + valid_fn)
    valid_loss = valid_loss / total_samples
    F_score = 2.0 * valid_precision * valid_recall / (valid_precision + valid_recall)
    return valid_loss, valid_accuracy, F_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="cnn", help="Model to train")
    parser.add_argument("-e", "--epochs", type=int, default=64, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-nf","--num_filters", type=int, default=32, help="Number of filters in the first convolutional layer")
    parser.add_argument("-fz","--filter_size", type=int, nargs="+", default=[2,3,4], help="Filter size in the cnn")
    parser.add_argument("-sl","--sequence_length", type=int, default=50, help="Sequence length")
    parser.add_argument("-t","--train", type=bool, default=True, help="Train the model")
    
    args = parser.parse_args()
    
    model_name = args.model
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_filters = args.num_filters
    filter_sizes = args.filter_size
    sequence_length = args.sequence_length

    # load the model
    if model_name == "cnn":
        model = CNN(num_filters, filter_sizes)

    # choose the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=============================')
    print('Model:         ', model_name)
    print('Batch size:    ', batch_size)
    print('Epoch:         ', num_epochs)
    print('Num filters:   ', num_filters)
    print('Sentence len:  ', sequence_length)
    print('Learning rate: ', learning_rate)
    print('Device:        ', device)
    print('Train Status:  ', args.train)
    print('=============================')


    path=os.getcwd()
    path = os.path.join(path, "Dataset")
    # load the data
    print("Loading data...")
    train_dataset = TextDataset(path,"train.txt", sequence_length)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = TextDataset(path,"validation.txt", sequence_length)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = TextDataset(path,"test.txt", sequence_length)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    print("Data loaded")
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5,min_lr=1e-5)

    # move the model to the device
    model.to(device)
    
    count = 0
    best_val_loss = float('inf')
    last_loss = float('inf')

    # train the model
    print("Training...")
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_F_score = train(model, train_loader, optimizer, criterion, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F-score: {train_F_score:.4f}")
        valid_loss, valid_accuracy, valid_F_score = validate(model, valid_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid F-score: {valid_F_score:.4f}")

        scheduler.step(valid_loss)
        last_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Learning Rate is now {last_lr}")

        if(valid_loss < best_val_loss):
            best_val_loss = valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        
        if abs(train_loss - last_loss) < 0.003:
            count += 1
            # Early Stopping
            if count > 10:
                break
        else:
            count = 0

        last_loss = train_loss

    # test the model
    print("Testing...")
    test_loss, test_accuracy, test_F_score = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F-score: {test_F_score:.4f}")
