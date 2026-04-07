"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 2 Part A entry point for loading the trained MNIST CNN,
 * printing the model, dumping the first convolutional-layer weights, and
 * visualizing the 10 conv1 filters.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import torch
from train_mnist import MyNetwork
from examine_network import save_first_layer_weights, visualize_first_layer_filters


# loads the saved MNIST network in evaluation mode
def load_saved_model(model_path='results/mnist_model.pth'):
    """Load the trained MNIST CNN from disk in evaluation mode.

    Args:
        model_path: Path of the saved model state dictionary.

    Returns:
        The trained ``MyNetwork`` model in evaluation mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


# main function
def main(argv):
    """Run Task 2 Part A and save the first-layer analysis artifacts.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model = load_saved_model()
    print('Model structure:')
    print(model)
    print()

    save_first_layer_weights(model)
    visualize_first_layer_filters(model)


if __name__ == "__main__":
    main(sys.argv)
