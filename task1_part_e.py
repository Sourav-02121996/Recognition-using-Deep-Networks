"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part E entry point for loading the saved MNIST model
 * in evaluation mode and running it on the first 10 examples from the test
 * set.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
from train_mnist import MyNetwork, MNIST_MEAN, MNIST_STD
from test_mnist import test_first_ten


# loads the saved MNIST model in evaluation mode
def load_saved_model(model_path='results/mnist_model.pth'):
    """Load the trained MNIST CNN from disk in evaluation mode.

    Args:
        model_path: Path of the saved model state dictionary.

    Returns:
        A tuple ``(model, device)`` ready for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No trained model found at {model_path}')

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded model from {model_path}')
    return model, device


# loads the MNIST test set without shuffling
def load_test_set(batch_size=1000):
    """Load the MNIST test split used for Task 1 Part E.

    Args:
        batch_size: Batch size used by the fixed-order test loader.

    Returns:
        A PyTorch data loader for the MNIST test split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './data',
            train=False,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=False
    )


# main function
def main(argv):
    """Run Task 1 Part E end to end.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model, device = load_saved_model()
    test_loader = load_test_set()
    test_first_ten(model, device, test_loader)


if __name__ == "__main__":
    main(sys.argv)
