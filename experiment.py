# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 5: Design your own experiment - evaluating network architecture variations

# import statements
import sys
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
import json
import time


# Configurable CNN network for experimentation
class ConfigurableCNN(nn.Module):
    """A CNN with configurable parameters for experimentation."""

    def __init__(self, num_filters1=10, num_filters2=20, filter_size=5,
                 fc_hidden=50, dropout_rate=0.5, pool_size=2):
        super(ConfigurableCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=filter_size)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.pool_size = pool_size

        # Calculate the flattened size after conv+pool layers
        # Input: 28x28
        # After conv1 (filter_size): 28-filter_size+1
        # After pool: (28-filter_size+1)//pool_size
        s1 = (28 - filter_size + 1) // pool_size
        # After conv2 (filter_size): s1-filter_size+1
        # After pool: (s1-filter_size+1)//pool_size
        s2 = (s1 - filter_size + 1) // pool_size
        flat_size = num_filters2 * s2 * s2

        self.fc1 = nn.Linear(flat_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)

    # computes a forward pass
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), self.pool_size))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), self.pool_size))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# trains and evaluates a single configuration
def evaluate_config(config, train_loader, test_loader, device, epochs=5):
    """Train and evaluate a single network configuration."""
    try:
        model = ConfigurableCNN(
            num_filters1=config['num_filters1'],
            num_filters2=config['num_filters2'],
            filter_size=config['filter_size'],
            fc_hidden=config['fc_hidden'],
            dropout_rate=config['dropout_rate'],
            pool_size=config['pool_size']
        ).to(device)
    except Exception as e:
        return {'error': str(e), 'test_accuracy': 0, 'train_time': 0}

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total

    total_params = sum(p.numel() for p in model.parameters())

    return {
        'test_accuracy': accuracy,
        'test_loss': avg_loss,
        'train_time': train_time,
        'total_params': total_params
    }


# runs the full experiment across three dimensions
def run_experiment(train_loader, test_loader, device):
    """Evaluate network variations along three dimensions using linear search."""
    # Base configuration
    base_config = {
        'num_filters1': 10,
        'num_filters2': 20,
        'filter_size': 5,
        'fc_hidden': 50,
        'dropout_rate': 0.5,
        'pool_size': 2
    }

    results = {}
    epochs = 5

    # Dimension 1: Number of convolution filters (num_filters1, num_filters2)
    print('\n=== Dimension 1: Number of Convolution Filters ===')
    filter_counts = [(5, 10), (10, 20), (15, 30), (20, 40), (30, 60), (40, 80)]
    dim1_results = []
    for f1, f2 in filter_counts:
        config = base_config.copy()
        config['num_filters1'] = f1
        config['num_filters2'] = f2
        label = f'({f1},{f2})'
        print(f'  Testing filters {label}...', end=' ', flush=True)
        result = evaluate_config(config, train_loader, test_loader, device, epochs)
        result['label'] = label
        dim1_results.append(result)
        print(f'Acc: {result["test_accuracy"]:.2f}%, Time: {result["train_time"]:.1f}s')
    results['filter_counts'] = dim1_results

    # Find best from dim1
    best_dim1 = max(dim1_results, key=lambda x: x['test_accuracy'])
    best_f1, best_f2 = filter_counts[dim1_results.index(best_dim1)]

    # Dimension 2: Dropout rate
    print('\n=== Dimension 2: Dropout Rate ===')
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dim2_results = []
    for dr in dropout_rates:
        config = base_config.copy()
        config['num_filters1'] = best_f1
        config['num_filters2'] = best_f2
        config['dropout_rate'] = dr
        label = f'{dr:.1f}'
        print(f'  Testing dropout={label}...', end=' ', flush=True)
        result = evaluate_config(config, train_loader, test_loader, device, epochs)
        result['label'] = label
        dim2_results.append(result)
        print(f'Acc: {result["test_accuracy"]:.2f}%, Time: {result["train_time"]:.1f}s')
    results['dropout_rates'] = dim2_results

    best_dim2 = max(dim2_results, key=lambda x: x['test_accuracy'])
    best_dr = dropout_rates[dim2_results.index(best_dim2)]

    # Dimension 3: Hidden layer size
    print('\n=== Dimension 3: Hidden Layer Size ===')
    hidden_sizes = [25, 50, 75, 100, 150, 200, 300]
    dim3_results = []
    for hs in hidden_sizes:
        config = base_config.copy()
        config['num_filters1'] = best_f1
        config['num_filters2'] = best_f2
        config['dropout_rate'] = best_dr
        config['fc_hidden'] = hs
        label = str(hs)
        print(f'  Testing fc_hidden={label}...', end=' ', flush=True)
        result = evaluate_config(config, train_loader, test_loader, device, epochs)
        result['label'] = label
        dim3_results.append(result)
        print(f'Acc: {result["test_accuracy"]:.2f}%, Time: {result["train_time"]:.1f}s')
    results['hidden_sizes'] = dim3_results

    return results


# plots the experiment results
def plot_results(results, save_path='results/experiment_results.png'):
    """Create a summary plot of the experiment results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dimension 1: Filter counts
    labels = [r['label'] for r in results['filter_counts']]
    accs = [r['test_accuracy'] for r in results['filter_counts']]
    times = [r['train_time'] for r in results['filter_counts']]
    ax = axes[0]
    color1 = 'tab:blue'
    ax.set_xlabel('Filter Counts (conv1, conv2)')
    ax.set_ylabel('Test Accuracy (%)', color=color1)
    ax.bar(range(len(labels)), accs, color=color1, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_title('Dim 1: Number of Filters')
    ax.set_ylim(95, 100)

    # Dimension 2: Dropout rates
    labels = [r['label'] for r in results['dropout_rates']]
    accs = [r['test_accuracy'] for r in results['dropout_rates']]
    ax = axes[1]
    ax.plot(labels, accs, 'b-o')
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Dim 2: Dropout Rate')
    ax.grid(True)

    # Dimension 3: Hidden sizes
    labels = [r['label'] for r in results['hidden_sizes']]
    accs = [r['test_accuracy'] for r in results['hidden_sizes']]
    ax = axes[2]
    ax.plot(labels, accs, 'b-o')
    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Dim 3: FC Hidden Nodes')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved experiment plot to {save_path}')


# main function
def main(argv):
    """Run the architecture experiment on Fashion MNIST."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(42)

    # Use Fashion MNIST for more room to see effects
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                          transform=transform),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                          transform=transform),
        batch_size=1000, shuffle=False)

    print('Using Fashion MNIST dataset for experimentation')
    print('Base config: filters=(10,20), filter_size=5, fc_hidden=50, dropout=0.5')
    print('Evaluating 3 dimensions with linear search strategy\n')

    # Hypotheses
    print('=== HYPOTHESES ===')
    print('Dim 1 (Filter counts): More filters should improve accuracy up to a')
    print('  point, then plateau. Training time should increase linearly.')
    print('Dim 2 (Dropout rate): Moderate dropout (0.3-0.5) should work best.')
    print('  Too low may overfit; too high may underfit.')
    print('Dim 3 (Hidden size): Larger hidden layers should improve accuracy')
    print('  initially, but very large layers may overfit or not help.')
    print()

    results = run_experiment(train_loader, test_loader, device)

    # Save raw results
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved raw results to results/experiment_results.json')

    # Plot results
    plot_results(results)

    # Summary
    print('\n=== SUMMARY ===')
    for dim_name, dim_key in [('Filter Counts', 'filter_counts'),
                               ('Dropout Rate', 'dropout_rates'),
                               ('Hidden Size', 'hidden_sizes')]:
        best = max(results[dim_key], key=lambda x: x['test_accuracy'])
        print(f'{dim_name}: Best config = {best["label"]} '
              f'with {best["test_accuracy"]:.2f}% accuracy')


if __name__ == "__main__":
    main(sys.argv)
