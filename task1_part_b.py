"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Standalone Task 1 Part B entry point for constructing the MNIST CNN,
 * printing the architecture, and verifying the output tensor shapes across
 * the network pipeline.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

# import statements
import os
import sys
import torch
import torch.nn.functional as F
from train_mnist import MyNetwork


# builds the CNN model required by Task 1 Part B
def build_cnn_model():
    """Create the MNIST CNN used throughout Task 1.

    Returns:
        An instance of ``MyNetwork``.
    """
    return MyNetwork()


# computes the tensor shapes after each major stage of the network
def get_network_shapes(model):
    """Record the output shapes after each major stage of the CNN.

    Args:
        model: The CNN model whose intermediate shapes should be verified.

    Returns:
        A list of ``(layer_name, shape)`` tuples describing the pipeline.
    """
    x = torch.zeros(1, 1, 28, 28)
    shapes = [('Input', tuple(x.shape[1:]))]

    x = model.conv1(x)
    shapes.append(('Conv2d(1, 10, 5x5)', tuple(x.shape[1:])))

    x = F.max_pool2d(x, 2)
    x = F.relu(x)
    shapes.append(('MaxPool2d(2x2) + ReLU', tuple(x.shape[1:])))

    x = model.conv2(x)
    shapes.append(('Conv2d(10, 20, 5x5)', tuple(x.shape[1:])))

    model.eval()
    x = model.conv2_drop(x)
    shapes.append(('Dropout2d(p=0.5)', tuple(x.shape[1:])))

    x = F.max_pool2d(x, 2)
    x = F.relu(x)
    shapes.append(('MaxPool2d(2x2) + ReLU', tuple(x.shape[1:])))

    x = x.view(-1, 320)
    shapes.append(('Flatten', tuple(x.shape[1:])))

    x = model.fc1(x)
    x = F.relu(x)
    shapes.append(('Linear(320, 50) + ReLU', tuple(x.shape[1:])))

    x = model.fc2(x)
    x = F.log_softmax(x, dim=1)
    shapes.append(('Linear(50, 10) + log_softmax', tuple(x.shape[1:])))

    return shapes


# writes the model printout and layer shapes to a text file
def save_network_summary(model, shapes, save_path='results/network_architecture.txt'):
    """Save the model printout and verified layer shapes to disk.

    Args:
        model: The CNN model to summarize.
        shapes: Verified output-shape tuples returned by ``get_network_shapes``.
        save_path: Output path for the text summary file.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = [
        'Task 1 Part B: CNN Network Model',
        '',
        'PyTorch model printout:',
        str(model),
        '',
        'Layer output shapes for a 1x28x28 MNIST input:'
    ]

    for layer_name, shape in shapes:
        lines.append(f'- {layer_name}: {shape}')

    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(lines) + '\n')

    print(f'Saved network summary to {save_path}')


# prints the model and the verified layer sequence
def print_network_summary(model, shapes):
    """Print the CNN architecture and the verified stage-by-stage shapes.

    Args:
        model: The CNN model to print.
        shapes: Verified output-shape tuples returned by ``get_network_shapes``.

    Returns:
        None.
    """
    print('Task 1 Part B: CNN Network Model\n')
    print(model)
    print('\nVerified layer sequence and output shapes:')
    for layer_name, shape in shapes:
        print(f'  {layer_name:<32} -> {shape}')


# main function
def main(argv):
    """Run Task 1 Part B end to end.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    model = build_cnn_model()
    shapes = get_network_shapes(model)
    print_network_summary(model, shapes)
    save_network_summary(model, shapes)


if __name__ == "__main__":
    main(sys.argv)
