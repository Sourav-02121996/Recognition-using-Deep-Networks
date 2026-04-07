"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part D entry point for ensuring the trained MNIST
 * network is saved to disk and can be loaded back into the CNN
 * architecture.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import json
import os
import sys
from datetime import datetime
import torch
from train_mnist import MyNetwork
from task1_part_c import run_training


# ensures the trained model file exists
def ensure_saved_model(model_path='results/mnist_model.pth'):
    """Ensure that the trained model file exists on disk.

    Args:
        model_path: Expected path of the saved model state dictionary.

    Returns:
        The resolved model path after creating the file if necessary.
    """
    if os.path.exists(model_path):
        print(f'Found existing saved model at {model_path}')
        return model_path

    print(f'No saved model found at {model_path}')
    print('Running Task 1 Part C training to create the saved model file.')
    run_training()
    return model_path


# loads the saved weights back into the network to verify the file
def verify_saved_model(model_path='results/mnist_model.pth'):
    """Load the saved weights file and record its metadata.

    Args:
        model_path: Path of the saved model state dictionary.

    Returns:
        A dictionary containing saved-model metadata and load status.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    file_stats = os.stat(model_path)
    model_info = {
        'model_path': model_path,
        'file_size_bytes': file_stats.st_size,
        'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(timespec='seconds'),
        'loaded_successfully': True
    }

    with open('results/model_file_info.json', 'w', encoding='utf-8') as output_file:
        json.dump(model_info, output_file, indent=2)

    print(f'Successfully loaded saved model from {model_path}')
    print(f"Model file size: {model_info['file_size_bytes']} bytes")
    print(f"Model file modified: {model_info['modified_time']}")
    print('Saved model metadata to results/model_file_info.json')
    return model_info


# main function
def main(argv):
    """Run Task 1 Part D by checking and validating the saved model file.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model_path = ensure_saved_model()
    verify_saved_model(model_path)


if __name__ == "__main__":
    main(sys.argv)
