"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Provides evaluation utilities for the saved MNIST model, including the
 * first-10 test-set inspection required by Task 1 Part E and the custom
 * handwritten-digit evaluation required by Task 1 Part F.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train_mnist import MyNetwork, MNIST_MEAN, MNIST_STD


# runs the model on the first 10 test examples and prints results
def test_first_ten(model, device, test_loader):
    """Evaluate the saved MNIST model on the first 10 test examples.

    Args:
        model: The trained MNIST CNN in evaluation mode.
        device: The torch device used for inference.
        test_loader: Fixed-order MNIST test data loader.

    Returns:
        A list of strings containing the printed output table.
    """
    os.makedirs('results', exist_ok=True)
    model.eval()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = model(example_data[:10].to(device))

    output_lines = []
    print('\n--- First 10 Test Examples ---')
    header = f'{"Idx":>3} | {"Network Output (10 values)":>60} | {"Pred":>4} | {"True":>4} | {"Match":>5}'
    separator = '-' * 85
    print(header)
    print(separator)
    output_lines.extend(['--- First 10 Test Examples ---', header, separator])
    for i in range(10):
        values = output[i].cpu().numpy()
        values_str = ' '.join([f'{v:6.2f}' for v in values])
        pred = output[i].argmax().item()
        true_label = example_targets[i].item()
        match = 'YES' if pred == true_label else 'NO'
        line = f'{i:3d} | {values_str} | {pred:4d} | {true_label:4d} | {match:>5}'
        print(line)
        output_lines.append(line)

    # Plot first 9 as 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(9):
        row, col = i // 3, i % 3
        display_image = example_data[i][0] * MNIST_STD + MNIST_MEAN
        axes[row][col].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
        pred = output[i].argmax().item()
        axes[row][col].set_title(f'Pred: {pred}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    plt.suptitle('First 9 Test Digits with Predictions')
    plt.tight_layout()
    plt.savefig('results/first_nine_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved first 9 predictions plot to results/first_nine_predictions.png')

    output_path = 'results/test_set_outputs.txt'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(output_lines) + '\n')
    print(f'Saved test-set output table to {output_path}')
    return output_lines


# tests the network on custom handwritten digit images
def test_custom_digits(model, device, image_dir='custom_digits'):
    """Evaluate the saved MNIST model on custom handwritten digit images.

    Args:
        model: The trained MNIST CNN in evaluation mode.
        device: The torch device used for inference.
        image_dir: Directory containing custom digit image files.

    Returns:
        A list of strings containing the printed custom-digit results, or
        ``None`` if no usable images are found.
    """
    if not os.path.exists(image_dir):
        print(f'\nNo custom digit directory found at {image_dir}')
        print('Please create handwritten digit images (0-9) and place them in this directory.')
        print('Name them 0.png, 1.png, ..., 9.png')
        return

    from PIL import Image, ImageOps

    def preprocess_custom_digit(path):
        """Convert a handwritten digit image into an MNIST-like tensor.

        Args:
            path: Filesystem path of the custom digit image.

        Returns:
            A normalized ``1 x 28 x 28`` tensor ready for inference.
        """
        img = Image.open(path).convert('L')
        img_array = np.array(img, dtype=np.uint8)

        # Match MNIST polarity: bright digit strokes on a dark background.
        if img_array.mean() > 127:
            img_array = 255 - img_array

        # Increase contrast so the dark marker strokes stand out after inversion.
        img_array = np.array(ImageOps.autocontrast(Image.fromarray(img_array, mode='L')), dtype=np.uint8)

        # Crop to the handwritten foreground so the digit fills the frame.
        foreground = np.argwhere(img_array > 20)
        if foreground.size > 0:
            top, left = foreground.min(axis=0)
            bottom, right = foreground.max(axis=0) + 1
            img_array = img_array[top:bottom, left:right]

        digit_image = Image.fromarray(img_array, mode='L')

        # Resize to fill a 24x24 box while preserving aspect ratio, then center on 28x28.
        target_size = 24
        width, height = digit_image.size
        scale = min(target_size / max(width, 1), target_size / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized_digit = digit_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (28, 28), color=0)
        offset_x = (28 - resized_width) // 2
        offset_y = (28 - resized_height) // 2
        canvas.paste(resized_digit, (offset_x, offset_y))

        img_tensor = transforms.ToTensor()(canvas)
        img_tensor = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))(img_tensor)
        return img_tensor

    results = []
    images = []
    result_lines = []
    source_paths = []

    for digit in range(10):
        # Try common extensions
        found = False
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']:
            path = os.path.join(image_dir, f'{digit}{ext}')
            if os.path.isfile(path):
                img_tensor = preprocess_custom_digit(path)
                images.append(img_tensor)
                results.append(digit)
                source_paths.append(path)
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
        line = f'Digit {true_digit}: Predicted {pred} {"CORRECT" if match else "WRONG"}'
        print(line)
        result_lines.append(line)

    accuracy_line = f'Accuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)'
    print(f'\n{accuracy_line}')
    result_lines.append('')
    result_lines.append(accuracy_line)

    # Plot original custom input images
    cols = min(5, len(source_paths))
    rows = (len(source_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    for i, path in enumerate(source_paths):
        row, col = i // cols, i % cols
        original_image = Image.open(path).convert('L')
        axes[row][col].imshow(original_image, cmap='gray')
        axes[row][col].set_title(f'Digit {results[i]}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    for i in range(len(source_paths), rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].set_visible(False)
    plt.suptitle('Original Custom Handwritten Digit Inputs')
    plt.tight_layout()
    plt.savefig('results/custom_digits_inputs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved original custom digit inputs to results/custom_digits_inputs.png')

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
        img_display = images[i][0].cpu().numpy() * MNIST_STD + MNIST_MEAN
        axes[row][col].imshow(img_display, cmap='gray', vmin=0.0, vmax=1.0)
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

    output_path = 'results/custom_digit_outputs.txt'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(result_lines) + '\n')
    print(f'Saved custom digit output table to {output_path}')
    return result_lines


# main function
def main(argv):
    """Run both the MNIST test-set evaluation and custom-digit evaluation.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
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
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
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
