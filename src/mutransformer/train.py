import torch
from torch import nn, optim

from mutransformer.utils import save_experiment, save_checkpoint
from mutransformer.dataset import prepare_data
from mutransformer.config import config
from mutransformer.model import MuTransformer

class Trainer:
    """
    The simple trainer.
    """
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        total_samples = 0  # To keep track of the total number of samples

        for batch in trainloader:
            # Unpack the batch
            (text_input_tensor, mubin_input_tensor), target_tensor = batch

            # Move the tensors to the device
            text_input_tensor = text_input_tensor.to(self.device)
            mubin_input_tensor = mubin_input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Calculate the loss
            loss = self.loss_fn(self.model(text_input_tensor, mubin_input_tensor), target_tensor)

            # Backpropagate the loss
            loss.backward()

            # Update the model's parameters
            self.optimizer.step()
            # Accumulate the total loss
            batch_size = text_input_tensor.size(0)  # Assuming batch size is the first dimension
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        # Compute the average loss over the epoch
        average_loss = total_loss / total_samples
        return average_loss

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0

        for batch in testloader:
            # Unpack the batch
            (text_input_tensor, binary_input_tensor), target_tensor = batch

            # Move the tensors to the device
            text_input_tensor = text_input_tensor.to(self.device)
            binary_input_tensor = binary_input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # Get predictions
            logits = self.model(text_input_tensor, binary_input_tensor)

            # Calculate the loss
            loss = self.loss_fn(logits, target_tensor)
            batch_size = text_input_tensor.size(0)
            total_loss += loss.item() * batch_size

            # Calculate the accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == target_tensor).item()

        # Calculate the accuracy and average loss
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)

        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    # Load the dataset

    data = prepare_data(batch_size=batch_size)
    if data is None:
        print("Data loading Failed")
        return
    trainloader, val_loader = data


    # Create the model, optimizer, loss function and trainer
    model = MuTransformer(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader, val_loader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)

if __name__ == "__main__":
    main()