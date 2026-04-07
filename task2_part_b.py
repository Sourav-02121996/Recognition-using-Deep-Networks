"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 2 Part B entry point for applying the learned conv1
 * filters to the first MNIST training example and saving the filtered-image
 * visualization.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from train_mnist import MNIST_MEAN, MNIST_STD
from examine_network import show_filter_effects
from task2_part_a import load_saved_model


# loads the MNIST training set without shuffling
def load_training_set(batch_size=64):
    """Load the MNIST training split in fixed order for filter inspection.

    Args:
        batch_size: Batch size used by the training data loader.

    Returns:
        A PyTorch data loader for the MNIST training split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=False
    )


# main function
def main(argv):
    """Run Task 2 Part B and save the filter-effect visualization.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model = load_saved_model()
    train_loader = load_training_set()
    first_label = show_filter_effects(model, train_loader)
    print(f'Applied conv1 filters to the first training example with label {first_label}.')


if __name__ == "__main__":
    main(sys.argv)
