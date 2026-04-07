"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Implements Task 3 transfer learning from the MNIST CNN to Greek letter
 * recognition for alpha, beta, and gamma.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from train_mnist import MyNetwork, MNIST_MEAN, MNIST_STD


# greek data set transform - converts RGB images to match MNIST format
class GreekTransform:
    """Transform Greek letter images into the MNIST-style input format."""
    def __init__(self):
        """Initialize the stateless Greek-image preprocessing transform."""
        pass

    def __call__(self, x):
        """Apply grayscale conversion, scaling, cropping, and inversion.

        Args:
            x: Input image tensor in RGB format.

        Returns:
            A transformed tensor shaped like an MNIST-style grayscale digit.
        """
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def preprocess_custom_greek_image(image):
    """Convert a photographed Greek-letter image into an MNIST-style tensor.

    Args:
        image: PIL image containing a handwritten Greek-letter example.

    Returns:
        A single-channel float tensor in the range ``[0, 1]`` with white
        foreground on a black background.
    """
    grayscale = image.convert('L')
    equalized = ImageOps.equalize(grayscale)
    resized = equalized.resize((128, 128), Image.Resampling.BILINEAR).convert('RGB')
    return GreekTransform()(transforms.ToTensor()(resized))


def resolve_greek_training_path():
    """Resolve the Greek-letter training directory from common local layouts.

    Returns:
        The filesystem path containing the ``alpha``, ``beta``, and ``gamma``
        subdirectories.

    Raises:
        FileNotFoundError: If no valid Greek training directory is found.
    """
    candidate_paths = [
        'greek_train',
        'data/greek_train'
    ]
    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        'Greek training data not found. Expected greek_train/ or data/greek_train/.'
    )


def save_modified_network_summary(model, save_path='results/greek_model_architecture.txt'):
    """Save the modified transfer-learning model printout to disk.

    Args:
        model: Transfer-learning model whose architecture should be saved.
        save_path: Output path for the architecture text file.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('Modified Network:\n')
        output_file.write(f'{model}\n')
    print(f'Saved modified network summary to {save_path}')


# trains the modified network on the greek letter dataset
def train_greek(model, device, greek_train, epochs=100, learning_rate=0.01):
    """Train the modified classifier on the Greek letter dataset.

    Args:
        model: Transfer-learning model with the 3-class output layer.
        device: The torch device used for training.
        greek_train: Data loader for the Greek training set.
        epochs: Maximum number of training epochs to run.
        learning_rate: SGD learning rate.

    Returns:
        A list containing the average training loss for each epoch run.
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    train_losses = []
    train_accuracies = []
    first_perfect_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in greek_train:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%')

        if accuracy >= 100.0 and first_perfect_epoch is None:
            first_perfect_epoch = epoch

    return {
        'epochs_run': len(train_losses),
        'first_perfect_epoch': first_perfect_epoch,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }


# tests the model on custom greek letter images
def test_custom_greek(model, device, image_dir='custom_greek'):
    """Evaluate the Greek-letter classifier on custom user images.

    Args:
        model: Trained Greek-letter classifier.
        device: The torch device used for inference.
        image_dir: Directory containing custom Greek-letter image files.

    Returns:
        None.
    """
    if not os.path.exists(image_dir):
        print(f'\nNo custom Greek letter directory found at {image_dir}')
        print('Please create alpha, beta, gamma images and place them in this directory.')
        return None

    normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    label_names = ['alpha', 'beta', 'gamma']
    images = []
    true_labels = []
    names = []

    def infer_label_from_name(fname_lower):
        """Infer the Greek-letter label from a filename.

        Args:
            fname_lower: Lowercase filename string.

        Returns:
            Integer class label for alpha, beta, or gamma, or ``-1`` if unknown.
        """
        if 'alpha' in fname_lower or 'aplha' in fname_lower:
            return 0
        if 'beta' in fname_lower:
            return 1
        if 'gamma' in fname_lower:
            return 2
        return -1

    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert('RGB')
            img_tensor = normalize(preprocess_custom_greek_image(img))
            images.append(img_tensor)
            fname_lower = fname.lower()
            true_labels.append(infer_label_from_name(fname_lower))
            names.append(fname)

    if len(images) == 0:
        print('No custom Greek images found.')
        return None

    batch = torch.stack(images).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print('\n--- Custom Greek Letter Results ---')
    correct = 0
    total = 0
    result_lines = []
    for i in range(len(images)):
        pred = output[i].argmax().item()
        true = true_labels[i]
        pred_name = label_names[pred]
        true_name = label_names[true] if true >= 0 else 'unknown'
        match = pred == true
        if true >= 0:
            correct += int(match)
            total += 1
        line = (f'{names[i]}: True={true_name}, Predicted={pred_name} '
                f'{"CORRECT" if match else "WRONG"}')
        print(line)
        result_lines.append(line)

    if total > 0:
        accuracy_line = f'Accuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)'
        print(f'\n{accuracy_line}')
        result_lines.append('')
        result_lines.append(accuracy_line)

    # Plot
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for i in range(len(images)):
        row, col = i // cols, i % cols
        display_image = images[i][0].cpu().numpy() * MNIST_STD + MNIST_MEAN
        axes[row][col].imshow(display_image, cmap='gray', vmin=0.0, vmax=1.0)
        pred = output[i].argmax().item()
        true = true_labels[i]
        if true >= 0:
            title = f'True: {label_names[true]}\nPred: {label_names[pred]}'
            color = 'green' if pred == true else 'red'
        else:
            title = f'Pred: {label_names[pred]}'
            color = 'black'
        axes[row][col].set_title(title, color=color)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    for i in range(len(images), rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].set_visible(False)
    plt.suptitle('Custom Greek Letter Classification')
    plt.tight_layout()
    plt.savefig('results/custom_greek_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved custom Greek results to results/custom_greek_results.png')
    with open('results/custom_greek_outputs.txt', 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(result_lines) + '\n')
    print('Saved custom Greek output table to results/custom_greek_outputs.txt')
    return {
        'num_images': len(images),
        'num_labeled_images': total,
        'accuracy': 100.0 * correct / total if total > 0 else None
    }


# main function
def main(argv):
    """Run the full Task 3 transfer-learning workflow.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs('results', exist_ok=True)

    # Generate the MNIST network and load pre-trained weights
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('results/mnist_model.pth',
                                     map_location=device, weights_only=True))
    print('Loaded pre-trained MNIST model')

    # Freeze all network weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer with a new Linear layer with 3 nodes
    model.fc2 = nn.Linear(50, 3).to(device)
    print('\nModified network (last layer replaced):')
    print(model)
    save_modified_network_summary(model)

    # Set up Greek letter data loader
    try:
        training_set_path = resolve_greek_training_path()
    except FileNotFoundError as exc:
        print(f'\nError: {exc}')
        return

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
            ])),
        batch_size=5,
        shuffle=True)

    # Print class mapping
    dataset = greek_train.dataset
    print(f'\nClass mapping: {dataset.class_to_idx}')
    print(f'Number of training samples: {len(dataset)}')

    # Train on Greek letters
    print('\nTraining on Greek letters...')
    training_history = train_greek(model, device, greek_train, epochs=200)

    # Plot training error
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(training_history['train_losses']) + 1),
            training_history['train_losses'], 'b-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Greek Letter Transfer Learning - Training Error')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('results/greek_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training loss plot to results/greek_training_loss.png')
    with open('results/greek_training_history.json', 'w', encoding='utf-8') as output_file:
        json.dump(training_history, output_file, indent=2)
    print('Saved Greek training history to results/greek_training_history.json')

    # Save the Greek model
    torch.save(model.state_dict(), 'results/greek_model.pth')
    print('Saved Greek model to results/greek_model.pth')

    # Test on custom Greek letters if available
    test_custom_greek(model, device)


if __name__ == "__main__":
    main(sys.argv)
