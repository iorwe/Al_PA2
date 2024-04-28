import torch

class EarlyStopping:
    def __init__(self, patience=10,  delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = float('inf')
        self.best_val_Fscore = 0.0
        self.best_val_accuracy = 0.0
        self.min_delta = delta
        self.save_path = path

    def __call__(self, val_loss, val_acc, val_Fscore, epoch, model):
        if val_loss < self.best_val_loss-self.min_delta or val_acc > self.best_val_accuracy or val_Fscore > self.best_val_Fscore:
            if val_loss < self.best_val_loss-self.min_delta:
                self.best_val_loss = val_loss
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                checkpoint = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, self.save_path)
            if val_Fscore > self.best_val_Fscore:
                self.best_val_Fscore = val_Fscore
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

