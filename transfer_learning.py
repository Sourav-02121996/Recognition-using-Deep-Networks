# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 3: Transfer learning on Greek letters (alpha, beta, gamma)

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train_mnist import MyNetwork


# greek data set transform - converts RGB images to match MNIST format
class GreekTransform:
    """Transform to convert Greek letter images to match MNIST digit format."""
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# trains the modified network on the greek letter dataset
def train_greek(model, device, greek_train, epochs=100, learning_rate=0.01):
    """Train the modified network on Greek letters until convergence."""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    train_losses = []

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

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%')

        # Stop if perfect accuracy
        if accuracy >= 100.0 and avg_loss < 0.01:
            print(f'Converged at epoch {epoch} with loss {avg_loss:.4f}')
            break

    return train_losses


# tests the model on custom greek letter images
def test_custom_greek(model, device, image_dir='custom_greek'):
    """Test the Greek letter classifier on custom images."""
    if not os.path.exists(image_dir):
        print(f'\nNo custom Greek letter directory found at {image_dir}')
        print('Please create alpha, beta, gamma images and place them in this directory.')
        return

    from PIL import Image

    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    label_names = ['alpha', 'beta', 'gamma']
    images = []
    true_labels = []
    names = []

    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            # Determine label from filename
            fname_lower = fname.lower()
            if 'alpha' in fname_lower:
                true_labels.append(0)
            elif 'beta' in fname_lower:
                true_labels.append(1)
            elif 'gamma' in fname_lower:
                true_labels.append(2)
            else:
                true_labels.append(-1)
            names.append(fname)

    if len(images) == 0:
        print('No custom Greek images found.')
        return

    batch = torch.stack(images).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print('\n--- Custom Greek Letter Results ---')
    correct = 0
    total = 0
    for i in range(len(images)):
        pred = output[i].argmax().item()
        true = true_labels[i]
        pred_name = label_names[pred]
        true_name = label_names[true] if true >= 0 else 'unknown'
        match = pred == true
        if true >= 0:
            correct += int(match)
            total += 1
        print(f'{names[i]}: True={true_name}, Predicted={pred_name} '
              f'{"CORRECT" if match else "WRONG"}')

    if total > 0:
        print(f'\nAccuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)')

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
        axes[row][col].imshow(images[i][0].cpu().numpy(), cmap='gray')
        pred = output[i].argmax().item()
        axes[row][col].set_title(f'Pred: {label_names[pred]}')
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


# main function
def main(argv):
    """Load pretrained MNIST model, modify for Greek letters, and train."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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

    # Set up Greek letter data loader
    training_set_path = 'greek_train'
    if not os.path.exists(training_set_path):
        print(f'\nError: Greek training data not found at {training_set_path}')
        print('Please download and extract the Greek letters dataset.')
        print('Expected structure:')
        print('  greek_train/')
        print('    alpha/')
        print('    beta/')
        print('    gamma/')
        return

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=5,
        shuffle=True)

    # Print class mapping
    dataset = greek_train.dataset
    print(f'\nClass mapping: {dataset.class_to_idx}')
    print(f'Number of training samples: {len(dataset)}')

    # Train on Greek letters
    print('\nTraining on Greek letters...')
    train_losses = train_greek(model, device, greek_train, epochs=200)

    # Plot training error
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Greek Letter Transfer Learning - Training Error')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('results/greek_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training loss plot to results/greek_training_loss.png')

    # Save the Greek model
    torch.save(model.state_dict(), 'results/greek_model.pth')
    print('Saved Greek model to results/greek_model.pth')

    # Test on custom Greek letters if available
    test_custom_greek(model, device)


if __name__ == "__main__":
    main(sys.argv)
