"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part C entry point for training the MNIST CNN,
 * collecting per-epoch metrics, and saving the training artifacts used in
 * the report.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import json
import sys
import torch
from train_mnist import MyNetwork, load_data, train_network


# trains the Task 1 CNN for at least five epochs
def run_training(epochs=5, batch_size_train=64, batch_size_test=1000):
    """Train the MNIST CNN for Task 1 Part C.

    Args:
        epochs: Number of training epochs to run.
        batch_size_train: Batch size for the training loader.
        batch_size_test: Batch size for the test loader.

    Returns:
        A dictionary containing the saved training history.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)
    train_loader, test_loader = load_data(batch_size_train, batch_size_test)
    model = MyNetwork().to(device)

    training_history = train_network(
        model,
        device,
        train_loader,
        test_loader,
        epochs=epochs
    )
    return training_history


# prints a compact summary of the training metrics
def print_training_summary(training_history):
    """Print a compact summary of the final saved training metrics.

    Args:
        training_history: Dictionary returned by ``run_training``.

    Returns:
        None.
    """
    final_epoch = training_history['epochs']
    final_train_loss = training_history['train_losses'][-1]
    final_test_loss = training_history['test_losses'][-1]
    final_train_accuracy = training_history['train_accuracies'][-1]
    final_test_accuracy = training_history['test_accuracies'][-1]

    print('\nTask 1 Part C summary:')
    print(f'  Final epoch: {final_epoch}')
    print(f'  Final train loss: {final_train_loss:.4f}')
    print(f'  Final test loss: {final_test_loss:.4f}')
    print(f'  Final train accuracy: {final_train_accuracy:.2f}%')
    print(f'  Final test accuracy: {final_test_accuracy:.2f}%')


# main function
def main(argv):
    """Run Task 1 Part C training and confirm the saved outputs.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    training_history = run_training()
    print_training_summary(training_history)

    with open('results/training_history.json', 'r', encoding='utf-8') as input_file:
        saved_history = json.load(input_file)
    print(f"  Saved {len(saved_history['train_losses'])} training epochs to results/training_history.json")


if __name__ == "__main__":
    main(sys.argv)
