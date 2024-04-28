import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm

from src.earlystopping import EarlyStopping
from src.dataset import TextDataset
from src.model.cnn import CNN
from src.model.rnn import RNN
from src.model.lstm import LSTM
from src.model.gru import GRU
from src.model.mlp import MLP

def train(model, train_loader, optimizer, criterion, device):
    # Set the model to training mode
    model.train()
    # Initialize the running loss
    train_loss = 0.
    train_tp, train_fp, train_tn, train_fn = 0.0, 0.0, 0.0, 0.0
    total_samples = 0.

    # Iterate over the training data
    for inputs, targets in tqdm.tqdm(train_loader,leave=False,desc='Training'):
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
        
def test(model, valid_loader, criterion, device,test=False):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the running loss
    valid_loss = 0.
    valid_tp, valid_fp, valid_tn, valid_fn = 0.0, 0.0, 0.0, 0.0
    total_samples = 0.

    with torch.no_grad():
        # Iterate over the validation data
        for inputs, targets in tqdm.tqdm(valid_loader,leave=False,desc=test*'Testing'+(not test)*'Validation'):
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
    # model
    parser.add_argument("-m", "--model", type=str, default="cnn", help="Model to train")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-sl","--sequence_length", type=int, default=50, help="Sequence length")
    parser.add_argument("-t","--train", type=str, default="Y", help="Train status: Y or N")

    # model cnn
    parser.add_argument("-nf","--num_filters", type=int, default=150, help="Number of filters in the first convolutional layer")
    parser.add_argument("-fz","--filter_size", type=int, nargs="+", default=[2,3,4], help="Filter size in the cnn")

    # model rnn, lstm, gru
    parser.add_argument("-hd","--hidden_dim", type=int, default=100, help="Hidden dimension in the rnn")
    parser.add_argument("-nl","--num_layers", type=int, default=1, help="Number of layers in the rnn")
    parser.add_argument("-bi","--bidirectional", type=bool, default=False, help="Bidirectional in the rnn")
    args = parser.parse_args()
    
    model_name = args.model
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    sequence_length = args.sequence_length
    if_train = args.train

    # model cnn
    num_filters = args.num_filters
    filter_sizes = args.filter_size

    # model rnn, lstm, gru
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    bidirectional=args.bidirectional

    # load the model
    if model_name == "cnn":
        model = CNN(num_filters, filter_sizes)
    elif model_name == "rnn":
        model = RNN(hidden_dim, num_layers, bidirectional)
    elif model_name == "lstm":
        model = LSTM(hidden_dim, num_layers, bidirectional)
    elif model_name == "gru":
        model = GRU(hidden_dim, num_layers, bidirectional)
    elif model_name == "mlp":
        model = MLP(sequence_length)
    else:
        print("!!! INVALID MODEL !!!")
        print("Please choose between 'cnn', 'rnn', 'lstm' and 'gru'")
        exit(1)
    # choose the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===================================================================================')
    print('Model:         ', model_name)
    print('Batch size:    ', batch_size)
    print('Epoch:         ', num_epochs)
    print('Sentence len:  ', sequence_length)
    print('Learning rate: ', learning_rate)
    print('Device:        ', device)
    print('Train Status:  ', if_train)
    if model_name == "cnn":
        print('Num filters:   ', num_filters)
        print('Filter sizes:  ', filter_sizes)
    elif model_name == "rnn" or model_name == "lstm" or model_name == "gru":
        print('Hidden dim:    ', hidden_dim)
        print('Num layers:    ', num_layers)
        print('Bidirectional: ', bidirectional)
    print('===================================================================================')

    # define the loss function
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5,min_lr=1e-5)

    # move the model to the device
    model.to(device)

    # create the directory to save the model
    os.makedirs("results", exist_ok=True)
    model_directory = os.path.join("results", model_name)
    os.makedirs(model_directory, exist_ok=True)
    if model_name == "cnn":
        model_filename = f"{model_name}_filters_{num_filters}x{filter_sizes}_length_{sequence_length}_batch_{batch_size}_lr_{learning_rate}.pth"
    elif model_name == "rnn" or model_name == "lstm" or model_name == "gru":
        model_filename = f"{model_name}_hidden_{hidden_dim}_layers_{num_layers}_bidirectional_{bidirectional}_length_{sequence_length}_batch_{batch_size}_lr_{learning_rate}.pth"
    else:
        model_filename = f"{model_name}_length_{sequence_length}_batch_{batch_size}_lr_{learning_rate}.pth"
    save_path = os.path.join(model_directory, model_filename)
    
    # check if the model exists
    if os.path.exists(save_path) and if_train == "Y":
        print("Model exists")
        print("Do you want to overwrite the model? (Y/N)")
        choice = input()

        if choice == "Y" or choice == "y":
            pass
        elif choice == "N" or choice == "n":
            if_train = "N"
        else:
            print("Invalid choice")
            exit(1)

    if if_train == "Y" or if_train == "y":
        path=os.getcwd()
        path = os.path.join(path, "Dataset")
        # load the data
        print("Loading data...")
        train_dataset = TextDataset(path,"train.txt", sequence_length)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        valid_dataset = TextDataset(path,"validation.txt", sequence_length)
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
        print("Data loaded")

        # train the model
        print("Start training...")
        print('===================================================================================')
        print('Epoch | Tr. Loss | Tr. Acc. | Tr. F-Score | Vaild Loss | Vaild Acc. | Vaild F-Score')
        print('------+----------+----------+-------------+------------+------------+--------------')

        # early stopping
        early_stopping = EarlyStopping(patience=10, delta=1e-4, path=save_path)

        for epoch in range(num_epochs):
            train_loss, train_accuracy, train_F_score = train(model, train_loader, optimizer, criterion, device)
            valid_loss, valid_accuracy, valid_F_score = test(model, valid_loader, criterion, device)
            print('{:5d} | {:8.4f} | {:8.4f} | {:11.4f} | {:10.4f} | {:10.4f} | {:11.4f}'.format(epoch+1, train_loss, train_accuracy, train_F_score, valid_loss, valid_accuracy, valid_F_score))
            # update the learning rate
            scheduler.step(valid_loss)
            # check early stopping
            early_stopping(valid_loss, valid_accuracy, valid_F_score, epoch, model)
            if early_stopping.early_stop:
                print('===================================================================================')
                print("!!! Early stopping at epoch ", epoch+1, " !!!")
                print("Best valid loss:     {:.4f}".format(early_stopping.best_val_loss))
                print("Best valid accuracy: {:.2%}".format(early_stopping.best_val_accuracy))
                print("Best valid F-Score:  {:.4f}".format(early_stopping.best_val_Fscore))
                print('===================================================================================')
                break

        print("Training finished")

    print("Start testing...")
    try:
        checkpoint = torch.load(save_path,map_location=device)
        model.load_state_dict(checkpoint['model'])
        print("Model loaded")
    except:
        print("!!! MODEL NOT FOUND !!!")
        print("Please train the model first")
        exit(1)

    path=os.getcwd()
    path = os.path.join(path, "Dataset")
    # load the test data
    test_dataset = TextDataset(path,"test.txt", sequence_length)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # test the model
    test_loss, test_accuracy, test_F_score = test(model, test_loader, criterion, device,test=True)
    print("Testing finished")
    print('===================================================================================')
    print('                        !!!!!!!  Test Results  !!!!!!!                             ')
    print('===================================================================================')
    print('Model:         {}'.format(model_name))
    print('Batch size:    {}'.format(batch_size))
    print('Epoch:         {}'.format(checkpoint['epoch']))
    print('Sentence len:  {}'.format(sequence_length))
    print('Learning rate: {}'.format(learning_rate))
    if model_name == "cnn":
        print('Num filters:   {}'.format(num_filters))
        print('Filter sizes:  {}'.format(filter_sizes))
    elif model_name == "rnn" or model_name == "lstm" or model_name == "gru":
        print('Hidden dim:    {}'.format(hidden_dim))
        print('Num layers:    {}'.format(num_layers))
        print('Bidirectional: {}'.format(bidirectional))
    print('Accuracy:      {:.2%}'.format(test_accuracy))
    print('F-Score:       {:.4f}'.format(test_F_score))
    print('===================================================================================')
