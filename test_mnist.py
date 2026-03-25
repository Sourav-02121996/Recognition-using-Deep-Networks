# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 1.5-1.6: Load trained network, test on MNIST and custom handwritten digits

# import statements
import sys
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train_mnist import MyNetwork


# runs the model on the first 10 test examples and prints results
def test_first_ten(model, device, test_loader):
    """Run model on first 10 test examples, print outputs and plot first 9."""
    model.eval()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = model(example_data[:10].to(device))

    print('\n--- First 10 Test Examples ---')
    print(f'{"Idx":>3} | {"Network Output (10 values)":>60} | {"Pred":>4} | {"True":>4} | {"Match":>5}')
    print('-' * 85)
    for i in range(10):
        values = output[i].cpu().numpy()
        values_str = ' '.join([f'{v:6.2f}' for v in values])
        pred = output[i].argmax().item()
        true_label = example_targets[i].item()
        match = 'YES' if pred == true_label else 'NO'
        print(f'{i:3d} | {values_str} | {pred:4d} | {true_label:4d} | {match:>5}')

    # Plot first 9 as 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(9):
        row, col = i // 3, i % 3
        axes[row][col].imshow(example_data[i][0], cmap='gray')
        pred = output[i].argmax().item()
        axes[row][col].set_title(f'Pred: {pred}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    plt.suptitle('First 9 Test Digits with Predictions')
    plt.tight_layout()
    plt.savefig('results/first_nine_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved first 9 predictions plot to results/first_nine_predictions.png')


# tests the network on custom handwritten digit images
def test_custom_digits(model, device, image_dir='custom_digits'):
    """Load custom digit images, preprocess, and classify."""
    if not os.path.exists(image_dir):
        print(f'\nNo custom digit directory found at {image_dir}')
        print('Please create handwritten digit images (0-9) and place them in this directory.')
        print('Name them 0.png, 1.png, ..., 9.png')
        return

    from PIL import Image

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    results = []
    images = []

    for digit in range(10):
        # Try common extensions
        found = False
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            path = os.path.join(image_dir, f'{digit}{ext}')
            if os.path.exists(path):
                img = Image.open(path)
                img_tensor = transform(img)

                # Invert if needed (MNIST is white on black)
                # Check mean intensity: if mostly white background, invert
                if img_tensor.mean() > 0.5:
                    img_tensor = 1.0 - img_tensor

                # Normalize with MNIST stats
                img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
                images.append(img_tensor)
                results.append(digit)
                found = True
                break

        if not found:
            print(f'Warning: No image found for digit {digit}')

    if len(images) == 0:
        print('No custom digit images found.')
        return

    # Stack and run through network
    batch = torch.stack(images).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print('\n--- Custom Handwritten Digit Results ---')
    correct = 0
    total = len(images)
    for i, true_digit in enumerate(results):
        pred = output[i].argmax().item()
        match = pred == true_digit
        correct += int(match)
        print(f'Digit {true_digit}: Predicted {pred} {"CORRECT" if match else "WRONG"}')

    print(f'\nAccuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)')

    # Plot results
    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    for i in range(len(images)):
        row, col = i // cols, i % cols
        # De-normalize for display
        img_display = images[i][0].cpu().numpy()
        axes[row][col].imshow(img_display, cmap='gray')
        pred = output[i].argmax().item()
        color = 'green' if pred == results[i] else 'red'
        axes[row][col].set_title(f'True: {results[i]}, Pred: {pred}', color=color)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    # Hide unused subplots
    for i in range(len(images), rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].set_visible(False)
    plt.suptitle('Custom Handwritten Digit Classification')
    plt.tight_layout()
    plt.savefig('results/custom_digits_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved custom digit results to results/custom_digits_results.png')


# main function
def main(argv):
    """Load trained model and run tests."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = MyNetwork().to(device)
    model_path = 'results/mnist_model.pth'
    if not os.path.exists(model_path):
        print(f'Error: No trained model found at {model_path}')
        print('Please run train_mnist.py first.')
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded model from {model_path}')

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=1000, shuffle=False)

    # Test on first 10 examples
    test_first_ten(model, device, test_loader)

    # Test on custom handwritten digits
    test_custom_digits(model, device)


if __name__ == "__main__":
    main(sys.argv)
