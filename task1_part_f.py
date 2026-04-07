"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part F entry point for loading the saved MNIST model
 * and classifying custom handwritten digit images provided by the user.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
from task1_part_e import load_saved_model
from test_mnist import test_custom_digits


# main function
def main(argv):
    """Run Task 1 Part F on the custom handwritten digit images.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model, device = load_saved_model()
    test_custom_digits(model, device)


if __name__ == "__main__":
    main(sys.argv)
