# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 1: Build and train a network to recognize MNIST digits

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# CNN network model for MNIST digit recognition
class MyNetwork(nn.Module):
    """CNN with two conv layers, dropout, and two fully connected layers."""

    def __init__(self):
        super(MyNetwork, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # A dropout layer with a 0.5 dropout rate
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # Fully connected layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)
        # Final fully connected layer with 10 nodes (one per digit)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        # Conv1 -> max pool 2x2 -> ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv2 -> dropout -> max pool 2x2 -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten
        x = x.view(-1, 320)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # FC2 -> log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# loads the MNIST training and test datasets
def load_data(batch_size_train=64, batch_size_test=1000):
    """Load MNIST train and test datasets with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader


# plots the first six examples from the test set
def plot_first_six(test_loader, save_path='results/first_six_digits.png'):
    """Plot the first six example digits from the test set."""
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i in range(6):
        row, col = i // 3, i % 3
        axes[row][col].imshow(example_data[i][0], cmap='gray')
        axes[row][col].set_title(f'Label: {example_targets[i].item()}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    plt.suptitle('First 6 Test Set Digits')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved first six digits plot to {save_path}')


# trains the network for one epoch
def train_epoch(model, device, train_loader, optimizer, epoch, train_losses):
    """Train for one epoch and record losses."""
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


# evaluates the network on the test set
def test_epoch(model, device, test_loader, test_losses):
    """Evaluate model on test set and record losses."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    test_losses.append(avg_loss)
    print(f'  Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


# trains the network for multiple epochs and saves results
def train_network(model, device, train_loader, test_loader, epochs=5,
                  learning_rate=0.01, momentum=0.5,
                  save_path='results/mnist_model.pth'):
    """Train network, plot loss curves, and save model."""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader,
                                            optimizer, epoch, train_losses)
        test_loss, test_acc = test_epoch(model, device, test_loader, test_losses)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plot training and testing error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), train_losses, 'b-o', label='Training Loss')
    ax1.plot(range(1, epochs + 1), test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), train_accuracies, 'b-o', label='Training Accuracy')
    ax2.plot(range(1, epochs + 1), test_accuracies, 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('results/training_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training plot to results/training_plot.png')

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Saved model to {save_path}')

    return train_losses, test_losses


# main function
def main(argv):
    """Main entry point: load data, train model, plot results."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load data
    batch_size_train = 64
    batch_size_test = 1000
    train_loader, test_loader = load_data(batch_size_train, batch_size_test)

    # Plot first six test examples
    plot_first_six(test_loader)

    # Build network
    model = MyNetwork().to(device)
    print('\nNetwork architecture:')
    print(model)

    # Train for 5 epochs
    epochs = 5
    train_network(model, device, train_loader, test_loader, epochs=epochs)

    print('\nTraining complete!')


if __name__ == "__main__":
    main(sys.argv)
