"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Implements Task 2 by loading the trained MNIST CNN, visualizing the
 * first convolutional-layer filters, and showing their effects on an input
 * digit.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from train_mnist import MyNetwork, MNIST_MEAN, MNIST_STD


# prints and saves the first-layer weight tensor information
def save_first_layer_weights(model, save_path='results/conv1_weights.txt'):
    """Print and save the conv1 weight tensor shape and filter values.

    Args:
        model: The trained MNIST CNN whose conv1 weights are inspected.
        save_path: Output path for the saved conv1 weight text dump.

    Returns:
        A torch tensor containing the conv1 weights on the CPU.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    weights = model.conv1.weight.detach().cpu()

    output_lines = [f'Conv1 weight shape: {list(weights.shape)}']
    print(output_lines[0])
    for i in range(weights.shape[0]):
        filter_weights = weights[i, 0].numpy()
        output_lines.append(f'\nFilter {i}:')
        output_lines.append(np.array2string(filter_weights, precision=6, suppress_small=False))
        print(f'Filter {i}:\n{filter_weights}\n')

    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(output_lines) + '\n')
    print(f'Saved conv1 weights to {save_path}')
    return weights


# visualizes the 10 filters of the first convolutional layer
def visualize_first_layer_filters(model, save_path='results/conv1_filters.png'):
    """Visualize and print the weights of the first convolutional layer.

    Args:
        model: The trained MNIST CNN whose first-layer filters are inspected.
        save_path: Output path for the saved filter visualization figure.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    weights = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i in range(10):
        row, col = i // 4, i % 4
        filt = weights[i, 0].numpy()
        axes[row][col].imshow(filt, cmap='gray')
        axes[row][col].set_title(f'Filter {i}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

    # Hide unused subplots
    for i in range(10, 12):
        row, col = i // 4, i % 4
        axes[row][col].set_visible(False)

    plt.suptitle('First Convolutional Layer Filters (conv1)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved filter visualization to {save_path}')


# applies the 10 filters to the first training example using OpenCV filter2D
def show_filter_effects(model, train_loader, save_path='results/filter_effects.png'):
    """Apply the first-layer filters to a sample training image.

    Args:
        model: The trained MNIST CNN whose filters are applied.
        train_loader: Training data loader used to fetch a sample digit.
        save_path: Output path for the saved filtered-image figure.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get first training example
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    first_image = example_data[0][0].numpy()  # 28x28
    first_label = example_targets[0].item()

    with torch.no_grad():
        weights = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(3, 4, figsize=(10, 7))

    # Show original image in first position
    display_image = first_image * MNIST_STD + MNIST_MEAN
    axes[0][0].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0][0].set_title(f'Original ({first_label})')
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    # Apply each filter
    for i in range(10):
        row, col = (i + 1) // 4, (i + 1) % 4
        kernel = weights[i, 0].numpy()
        filtered = cv2.filter2D(first_image, -1, kernel)
        axes[row][col].imshow(filtered, cmap='gray')
        axes[row][col].set_title(f'Filter {i}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

    # Hide last unused subplot
    axes[2][3].set_visible(False)

    plt.suptitle('Effect of Conv1 Filters on First Training Example')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved filter effects to {save_path}')
    return first_label


# main function
def main(argv):
    """Run Task 2 by loading the model and producing filter diagnostics.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('results/mnist_model.pth',
                                     map_location=device, weights_only=True))
    model.eval()

    # Print model structure
    print('Model structure:')
    print(model)
    print()

    # Save and print first-layer weights
    save_first_layer_weights(model)

    # Visualize first layer filters
    visualize_first_layer_filters(model)

    # Load training data for filter effects
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=64, shuffle=False)

    # Show filter effects
    show_filter_effects(model, train_loader)


if __name__ == "__main__":
    main(sys.argv)
