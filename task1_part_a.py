"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part A entry point for loading the MNIST test split
 * and saving the first-six-digit preview required in the report.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from train_mnist import plot_first_six, MNIST_MEAN, MNIST_STD


# loads the MNIST test dataset without shuffling
def load_mnist_test_data(batch_size=6):
    """Load the MNIST test split used for the Task 1 Part A preview.

    Args:
        batch_size: Batch size used for the fixed-order test loader.

    Returns:
        A tuple ``(test_dataset, test_loader)`` for the MNIST test split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return test_dataset, test_loader


# main function
def main(argv):
    """Run Task 1 Part A and save the required test-digit preview.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    test_dataset, test_loader = load_mnist_test_data()
    print(f'Loaded MNIST test set with {len(test_dataset)} images.')
    plot_first_six(test_loader)


if __name__ == "__main__":
    main(sys.argv)
