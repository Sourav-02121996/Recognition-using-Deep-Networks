"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Implements the core MNIST convolutional neural network, dataset loading,
 * training loop, evaluation loop, and training artifact generation used by
 * Task 1 of the project.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# CNN network model for MNIST digit recognition
class MyNetwork(nn.Module):
    """MNIST convolutional classifier with two conv blocks and two dense layers."""

    def __init__(self):
        """Initialize the convolutional, dropout, and linear layers."""
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
        """Run a forward pass through the CNN.

        Args:
            x: Input tensor of shape ``[batch_size, 1, 28, 28]``.

        Returns:
            A tensor of log-probabilities with shape ``[batch_size, 10]``.
        """
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
    """Load the MNIST train and test data loaders.

    Args:
        batch_size_train: Batch size used for the shuffled training loader.
        batch_size_test: Batch size used for the fixed-order test loader.

    Returns:
        A tuple ``(train_loader, test_loader)`` of PyTorch data loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
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
    """Save a 2x3 plot of the first six MNIST test digits.

    Args:
        test_loader: Data loader for the MNIST test split in fixed order.
        save_path: Output path for the saved figure.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i in range(6):
        row, col = i // 3, i % 3
        display_image = example_data[i][0] * MNIST_STD + MNIST_MEAN
        axes[row][col].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
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
    """Train the network for one epoch and record aggregate metrics.

    Args:
        model: The CNN model being trained.
        device: The torch device on which training is performed.
        train_loader: Training data loader.
        optimizer: Optimizer used to update model parameters.
        epoch: One-based epoch index used for logging.
        train_losses: List that is updated in place with the average epoch loss.

    Returns:
        A tuple ``(avg_loss, accuracy)`` for the completed epoch.
    """
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
    """Evaluate the network on the MNIST test set.

    Args:
        model: The CNN model being evaluated.
        device: The torch device on which evaluation is performed.
        test_loader: Test data loader.
        test_losses: List that is updated in place with the average test loss.

    Returns:
        A tuple ``(avg_loss, accuracy)`` for the evaluation pass.
    """
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
    """Train the CNN, save plots and metrics, and persist the model weights.

    Args:
        model: The CNN model to train.
        device: The torch device on which training is performed.
        train_loader: Training data loader.
        test_loader: Test data loader.
        epochs: Number of training epochs to run.
        learning_rate: SGD learning rate.
        momentum: SGD momentum factor.
        save_path: Output path for the saved model state dictionary.

    Returns:
        A dictionary containing the saved training and evaluation metrics.
    """
    os.makedirs('results', exist_ok=True)
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

    training_history = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
    with open('results/training_history.json', 'w', encoding='utf-8') as output_file:
        json.dump(training_history, output_file, indent=2)
    print('Saved training history to results/training_history.json')

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Saved model to {save_path}')

    return training_history


# main function
def main(argv):
    """Run the full Task 1 MNIST training pipeline.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
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
