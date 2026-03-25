# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 2: Examine the trained network - analyze filters and their effects

# import statements
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from train_mnist import MyNetwork


# visualizes the 10 filters of the first convolutional layer
def visualize_first_layer_filters(model, save_path='results/conv1_filters.png'):
    """Get and visualize the 10 5x5 filters from conv1."""
    weights = model.conv1.weight.detach().cpu()
    print(f'Conv1 weight shape: {weights.shape}')  # [10, 1, 5, 5]

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i in range(10):
        row, col = i // 4, i % 4
        filt = weights[i, 0].numpy()
        axes[row][col].imshow(filt, cmap='gray')
        axes[row][col].set_title(f'Filter {i}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        print(f'Filter {i}:\n{filt}\n')

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
    """Apply conv1 filters to the first training image and visualize."""
    # Get first training example
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    first_image = example_data[0][0].numpy()  # 28x28

    with torch.no_grad():
        weights = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(3, 4, figsize=(10, 7))

    # Show original image in first position
    axes[0][0].imshow(first_image, cmap='gray')
    axes[0][0].set_title('Original')
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


# main function
def main(argv):
    """Load trained model and examine its structure and filters."""
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
