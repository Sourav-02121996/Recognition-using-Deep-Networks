"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Implements Task 5 by running an assignment-scale architecture search on
 * Fashion MNIST and summarizing how three CNN dimensions affect accuracy,
 * training time, and parameter count.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

import json
import os
import random
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


FASHION_MEAN = 0.2860
FASHION_STD = 0.3530


class ConfigurableCNN(nn.Module):
    """CNN variant whose architecture can be adjusted for Task 5 sweeps."""

    def __init__(self, num_filters1=10, num_filters2=20, filter_size=5,
                 fc_hidden=50, dropout_rate=0.5, pool_size=2):
        """Initialize a configurable CNN architecture.

        Args:
            num_filters1: Filter count for the first convolution.
            num_filters2: Filter count for the second convolution.
            filter_size: Shared kernel size for both convolutions.
            fc_hidden: Hidden-node count of the dense layer.
            dropout_rate: Dropout probability after the second convolution.
            pool_size: Pooling window size used after each convolution.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=filter_size)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.pool_size = pool_size

        size_after_conv1 = (28 - filter_size + 1) // pool_size
        size_after_conv2 = (size_after_conv1 - filter_size + 1) // pool_size
        flat_size = num_filters2 * size_after_conv2 * size_after_conv2

        self.fc1 = nn.Linear(flat_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)

    def forward(self, x):
        """Run a forward pass through the configurable CNN.

        Args:
            x: Input tensor of shape ``[batch_size, 1, 28, 28]``.

        Returns:
            Log-probabilities with shape ``[batch_size, 10]``.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), self.pool_size))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), self.pool_size))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def set_seed(seed):
    """Seed the random number generators used by the experiment.

    Args:
        seed: Integer random seed value.

        Returns:
            None.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred_device='mps'):
    """Resolve the best available training device for Task 5.

    Args:
        preferred_device: Preferred device string for local acceleration.

    Returns:
        A ``torch.device`` instance available in the current environment.
    """
    if preferred_device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if preferred_device == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_fashion_datasets(seed, train_subset_size):
    """Create Fashion MNIST datasets and the fixed search subset.

    Args:
        seed: Seed used to select the subset indices reproducibly.
        train_subset_size: Number of training images to include in the search
            subset.

    Returns:
        A tuple ``(search_subset, full_train_dataset, test_dataset)``.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MEAN,), (FASHION_STD,))
    ])

    full_train_dataset = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(full_train_dataset), generator=generator)
    subset_indices = permutation[:train_subset_size].tolist()
    search_subset = torch.utils.data.Subset(full_train_dataset, subset_indices)
    return search_subset, full_train_dataset, test_dataset


def build_loader(dataset, batch_size, shuffle):
    """Create a data loader for a dataset.

    Args:
        dataset: Dataset or subset to iterate over.
        batch_size: Batch size used by the loader.
        shuffle: Whether to shuffle the data order.

    Returns:
        A ``DataLoader`` instance for the provided dataset.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def evaluate_config(config, train_loader, test_loader, device, epochs):
    """Train and evaluate one CNN configuration.

    Args:
        config: Dictionary describing the CNN hyperparameters.
        train_loader: Training data loader.
        test_loader: Test data loader.
        device: Torch device used for training and evaluation.
        epochs: Number of training epochs for this run.

    Returns:
        A dictionary containing accuracy, loss, timing, and parameter metrics.
    """
    try:
        model = ConfigurableCNN(
            num_filters1=config['num_filters1'],
            num_filters2=config['num_filters2'],
            filter_size=config['filter_size'],
            fc_hidden=config['fc_hidden'],
            dropout_rate=config['dropout_rate'],
            pool_size=config['pool_size']
        ).to(device)
    except Exception as exc:
        return {
            'error': str(exc),
            'test_accuracy': 0.0,
            'test_loss': float('inf'),
            'train_time': 0.0,
            'total_params': 0
        }

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    start_time = time.time()
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    return {
        'test_accuracy': 100.0 * correct / total,
        'test_loss': test_loss / total,
        'train_time': train_time,
        'total_params': sum(param.numel() for param in model.parameters())
    }


def summarize_config(config):
    """Convert a CNN configuration into a short human-readable label.

    Args:
        config: Dictionary describing one CNN architecture.

    Returns:
        A compact string label for logging and saved results.
    """
    return (
        f"filters=({config['num_filters1']},{config['num_filters2']}), "
        f"dropout={config['dropout_rate']:.2f}, hidden={config['fc_hidden']}"
    )


def run_sweep(name, stage, values, base_config, updater, labeler,
              train_loader, test_loader, device, epochs):
    """Evaluate one search sweep across a single architectural dimension.

    Args:
        name: Dimension name, such as ``filter_counts``.
        stage: Search stage label, such as ``coarse`` or ``refine``.
        values: Iterable of candidate values for this sweep.
        base_config: Starting architecture dictionary.
        updater: Function that applies one candidate value to a config copy.
        labeler: Function that converts a candidate value to a display label.
        train_loader: Training data loader.
        test_loader: Test data loader.
        device: Torch device used for training and evaluation.
        epochs: Number of epochs per configuration.

    Returns:
        A tuple ``(results, best_value, best_result)`` for the sweep.
    """
    results = []
    print(f'\n=== {name} ({stage}) ===')

    for index, value in enumerate(values, start=1):
        config = base_config.copy()
        updater(config, value)
        label = labeler(value)
        print(f'  [{index}/{len(values)}] Testing {label}...', end=' ', flush=True)
        metrics = evaluate_config(config, train_loader, test_loader, device, epochs)
        metrics['label'] = label
        metrics['value'] = value
        metrics['config'] = config
        results.append(metrics)
        print(
            f'Acc: {metrics["test_accuracy"]:.2f}%, '
            f'Loss: {metrics["test_loss"]:.4f}, '
            f'Time: {metrics["train_time"]:.1f}s'
        )

    best_result = max(results, key=lambda item: item['test_accuracy'])
    best_value = best_result['value']
    print(
        f'Best {name} {stage}: {best_result["label"]} '
        f'with {best_result["test_accuracy"]:.2f}% accuracy'
    )
    return results, best_value, best_result


def build_search_plan():
    """Create the Task 5 two-pass linear-search plan.

    Args:
        None.

    Returns:
        A dictionary containing hypotheses and candidate ranges.
    """
    return {
        'hypotheses': {
            'filters': (
                'Increasing convolution filter counts should improve Fashion '
                'MNIST accuracy up to a plateau, while training time and '
                'parameter count should grow steadily.'
            ),
            'dropout': (
                'Moderate dropout should perform best. Very low dropout will '
                'risk overfitting, while very high dropout will underfit.'
            ),
            'hidden': (
                'Increasing the fully connected hidden size should help at '
                'first, but very large hidden layers should show diminishing '
                'returns relative to their extra parameter cost.'
            ),
        },
        'coarse': {
            'filters': [4, 6, 8, 10, 12, 16, 20, 24, 32],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'hidden': [16, 24, 32, 48, 64, 96, 128, 192, 256],
        }
    }


def build_filter_refine_values(best_value):
    """Generate refined filter-count candidates around the best coarse value.

    Args:
        best_value: Best first-layer filter count from the coarse sweep.

    Returns:
        A sorted list of refined first-layer filter counts.
    """
    offsets = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    values = sorted({max(4, best_value + offset) for offset in offsets})
    return values


def build_dropout_refine_values(best_value):
    """Generate refined dropout candidates around the best coarse value.

    Args:
        best_value: Best dropout rate from the coarse sweep.

    Returns:
        A sorted list of refined dropout rates.
    """
    values = []
    for step in range(-4, 5):
        candidate = max(0.0, min(0.8, best_value + 0.05 * step))
        values.append(round(candidate, 2))
    return sorted(set(values))


def build_hidden_refine_values(best_value):
    """Generate refined hidden-size candidates around the best coarse value.

    Args:
        best_value: Best dense hidden size from the coarse sweep.

    Returns:
        A sorted list of refined hidden sizes.
    """
    candidates = {
        max(8, best_value // 2),
        max(8, int(round(best_value * 0.75))),
        max(8, best_value - 24),
        max(8, best_value - 16),
        max(8, best_value - 8),
        best_value,
        best_value + 8,
        best_value + 16,
        int(round(best_value * 1.25)),
        int(round(best_value * 1.5)),
    }
    return sorted(candidates)


def run_experiment(search_train_loader, full_train_loader, test_loader, device):
    """Run the full Task 5 experiment plan and final confirmation runs.

    Args:
        search_train_loader: Training loader for the search subset.
        full_train_loader: Training loader for the full Fashion MNIST dataset.
        test_loader: Test loader for Fashion MNIST.
        device: Torch device used for training and evaluation.

    Returns:
        A dictionary containing the experiment plan, raw sweep data, and final
        confirmation metrics.
    """
    base_config = {
        'num_filters1': 10,
        'num_filters2': 20,
        'filter_size': 5,
        'fc_hidden': 50,
        'dropout_rate': 0.5,
        'pool_size': 2
    }
    search_epochs = 2
    confirm_epochs = 5
    plan = build_search_plan()
    results = {
        'plan': plan,
        'base_config': base_config.copy(),
        'search_epochs': search_epochs,
        'confirmation_epochs': confirm_epochs,
        'dimensions': {},
    }

    total_evaluations = 0

    filter_coarse, best_filter_value, _ = run_sweep(
        name='Filter Counts',
        stage='coarse',
        values=plan['coarse']['filters'],
        base_config=base_config,
        updater=lambda config, value: config.update({
            'num_filters1': value,
            'num_filters2': value * 2
        }),
        labeler=lambda value: f'({value},{value * 2})',
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(filter_coarse)

    filter_refine_values = build_filter_refine_values(best_filter_value)
    filter_refine, _, filter_refine_best = run_sweep(
        name='Filter Counts',
        stage='refine',
        values=filter_refine_values,
        base_config=base_config,
        updater=lambda config, value: config.update({
            'num_filters1': value,
            'num_filters2': value * 2
        }),
        labeler=lambda value: f'({value},{value * 2})',
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(filter_refine)
    best_filter_result = max(filter_coarse + filter_refine,
                             key=lambda item: item['test_accuracy'])
    best_filter_value = best_filter_result['value']
    base_config['num_filters1'] = best_filter_value
    base_config['num_filters2'] = best_filter_value * 2

    dropout_coarse, best_dropout_value, _ = run_sweep(
        name='Dropout Rate',
        stage='coarse',
        values=plan['coarse']['dropout'],
        base_config=base_config,
        updater=lambda config, value: config.update({'dropout_rate': value}),
        labeler=lambda value: f'{value:.2f}',
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(dropout_coarse)

    dropout_refine_values = build_dropout_refine_values(best_dropout_value)
    dropout_refine, _, dropout_refine_best = run_sweep(
        name='Dropout Rate',
        stage='refine',
        values=dropout_refine_values,
        base_config=base_config,
        updater=lambda config, value: config.update({'dropout_rate': value}),
        labeler=lambda value: f'{value:.2f}',
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(dropout_refine)
    best_dropout_result = max(dropout_coarse + dropout_refine,
                              key=lambda item: item['test_accuracy'])
    best_dropout_value = best_dropout_result['value']
    base_config['dropout_rate'] = best_dropout_value

    hidden_coarse, best_hidden_value, _ = run_sweep(
        name='Hidden Layer Size',
        stage='coarse',
        values=plan['coarse']['hidden'],
        base_config=base_config,
        updater=lambda config, value: config.update({'fc_hidden': value}),
        labeler=lambda value: str(value),
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(hidden_coarse)

    hidden_refine_values = build_hidden_refine_values(best_hidden_value)
    hidden_refine, _, hidden_refine_best = run_sweep(
        name='Hidden Layer Size',
        stage='refine',
        values=hidden_refine_values,
        base_config=base_config,
        updater=lambda config, value: config.update({'fc_hidden': value}),
        labeler=lambda value: str(value),
        train_loader=search_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=search_epochs
    )
    total_evaluations += len(hidden_refine)
    best_hidden_result = max(hidden_coarse + hidden_refine,
                             key=lambda item: item['test_accuracy'])
    best_hidden_value = best_hidden_result['value']
    base_config['fc_hidden'] = best_hidden_value

    results['dimensions']['filter_counts'] = {
        'coarse': filter_coarse,
        'refine': filter_refine,
        'best': best_filter_result
    }
    results['dimensions']['dropout_rates'] = {
        'coarse': dropout_coarse,
        'refine': dropout_refine,
        'best': best_dropout_result
    }
    results['dimensions']['hidden_sizes'] = {
        'coarse': hidden_coarse,
        'refine': hidden_refine,
        'best': best_hidden_result
    }

    print('\n=== Confirmation Runs On Full Fashion MNIST Training Set ===')
    baseline_full = evaluate_config(
        results['base_config'],
        full_train_loader,
        test_loader,
        device,
        confirm_epochs
    )
    baseline_full['label'] = summarize_config(results['base_config'])
    print(
        '  Baseline full-data run: '
        f'{baseline_full["test_accuracy"]:.2f}% accuracy, '
        f'{baseline_full["test_loss"]:.4f} loss'
    )

    best_config = base_config.copy()
    best_full = evaluate_config(
        best_config,
        full_train_loader,
        test_loader,
        device,
        confirm_epochs
    )
    best_full['label'] = summarize_config(best_config)
    print(
        '  Best full-data run: '
        f'{best_full["test_accuracy"]:.2f}% accuracy, '
        f'{best_full["test_loss"]:.4f} loss'
    )

    total_evaluations += 2
    results['search_evaluations'] = total_evaluations - 2
    results['final_best_config'] = best_config
    results['final_best_config_label'] = summarize_config(best_config)
    results['full_data_confirmation'] = {
        'baseline': baseline_full,
        'best_config': best_full
    }
    results['total_evaluations'] = total_evaluations
    return results


def flatten_dimension_results(dimension_results):
    """Flatten coarse and refine sweep results into one ordered series.

    Args:
        dimension_results: Dictionary containing ``coarse`` and ``refine``
            result lists for one dimension.

    Returns:
        A single ordered list of sweep results.
    """
    series = []
    for stage in ['coarse', 'refine']:
        prefix = 'C' if stage == 'coarse' else 'R'
        for item in dimension_results[stage]:
            copied = item.copy()
            copied['plot_label'] = f'{prefix}:{item["label"]}'
            series.append(copied)
    return series


def plot_dimension(ax, items, title):
    """Plot one dimension's sweep results on a single subplot.

    Args:
        ax: Matplotlib axis to draw on.
        items: Ordered sweep-result list for one dimension.
        title: Title of the subplot.

    Returns:
        None.
    """
    x_positions = list(range(len(items)))
    accuracies = [item['test_accuracy'] for item in items]
    labels = [item['plot_label'] for item in items]

    coarse_positions = [i for i, item in enumerate(items) if item['plot_label'].startswith('C:')]
    refine_positions = [i for i, item in enumerate(items) if item['plot_label'].startswith('R:')]
    coarse_values = [accuracies[i] for i in coarse_positions]
    refine_values = [accuracies[i] for i in refine_positions]

    ax.plot(coarse_positions, coarse_values, 'o-', label='Coarse Sweep')
    ax.plot(refine_positions, refine_values, 's-', label='Refine Sweep')

    best_index = max(range(len(items)), key=lambda index: accuracies[index])
    ax.scatter(best_index, accuracies[best_index], color='red', s=60, zorder=5)
    ax.annotate(
        f'Best {accuracies[best_index]:.2f}%',
        (best_index, accuracies[best_index]),
        textcoords='offset points',
        xytext=(0, 8),
        ha='center',
        fontsize=9
    )

    ax.set_title(title)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend()


def save_summary_text(results, save_path):
    """Write a compact text summary of Task 5 to disk.

    Args:
        results: Full Task 5 results dictionary.
        save_path: Output path for the summary text file.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    baseline = results['full_data_confirmation']['baseline']
    best = results['full_data_confirmation']['best_config']
    best_config = results['final_best_config']

    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('Task 5 Experiment Summary\n')
        output_file.write(f'Search evaluations: {results["search_evaluations"]}\n')
        output_file.write(f'Total evaluations: {results["total_evaluations"]}\n')
        output_file.write(f'Final best config: {results["final_best_config_label"]}\n')
        output_file.write(
            'Baseline full-data accuracy: '
            f'{baseline["test_accuracy"]:.2f}%\n'
        )
        output_file.write(
            'Best full-data accuracy: '
            f'{best["test_accuracy"]:.2f}%\n'
        )
        output_file.write(
            'Full-data accuracy gain over baseline: '
            f'{best["test_accuracy"] - baseline["test_accuracy"]:.2f} points\n'
        )
        output_file.write('\nBest architecture values:\n')
        output_file.write(f'  Filters: ({best_config["num_filters1"]}, {best_config["num_filters2"]})\n')
        output_file.write(f'  Dropout: {best_config["dropout_rate"]:.2f}\n')
        output_file.write(f'  Hidden size: {best_config["fc_hidden"]}\n')


def plot_results(results, save_path='results/experiment_results.png'):
    """Create and save the Task 5 experiment summary plot.

    Args:
        results: Results dictionary returned by ``run_experiment``.
        save_path: Output path for the figure.

    Returns:
        None.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    plot_dimension(
        axes[0],
        flatten_dimension_results(results['dimensions']['filter_counts']),
        'Task 5 Dimension 1: Convolution Filter Counts'
    )
    plot_dimension(
        axes[1],
        flatten_dimension_results(results['dimensions']['dropout_rates']),
        'Task 5 Dimension 2: Dropout Rate'
    )
    plot_dimension(
        axes[2],
        flatten_dimension_results(results['dimensions']['hidden_sizes']),
        'Task 5 Dimension 3: Hidden Layer Size'
    )
    axes[2].set_xlabel('Sweep Order (C = coarse, R = refine)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(argv):
    """Run the full Task 5 architecture experiment on Fashion MNIST.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    del argv
    seed = 42
    train_subset_size = 12000
    search_batch_size = 128
    full_batch_size = 128

    set_seed(seed)
    device = resolve_device('mps')
    os.makedirs('results', exist_ok=True)
    print(f'Using device: {device}')

    search_subset, full_train_dataset, test_dataset = build_fashion_datasets(
        seed=seed,
        train_subset_size=train_subset_size
    )
    search_train_loader = build_loader(search_subset, search_batch_size, shuffle=True)
    full_train_loader = build_loader(full_train_dataset, full_batch_size, shuffle=True)
    test_loader = build_loader(test_dataset, 1000, shuffle=False)

    print('Using Fashion MNIST dataset for experimentation')
    print(f'Search subset size: {train_subset_size} / {len(full_train_dataset)}')
    print('Task 5 plan: assignment-scale two-pass linear search with 50+ runs')

    results = run_experiment(search_train_loader, full_train_loader, test_loader, device)
    results['dataset'] = {
        'name': 'FashionMNIST',
        'train_subset_size': train_subset_size,
        'full_train_size': len(full_train_dataset),
        'test_size': len(test_dataset)
    }
    results['device'] = str(device)
    results['seed'] = seed

    with open('results/experiment_results.json', 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=2)
    print('Saved raw results to results/experiment_results.json')

    plot_results(results, save_path='results/experiment_results.png')
    print('Saved experiment plot to results/experiment_results.png')

    save_summary_text(results, 'results/experiment_summary.txt')
    print('Saved experiment summary to results/experiment_summary.txt')

    best = results['full_data_confirmation']['best_config']
    baseline = results['full_data_confirmation']['baseline']
    print('\n=== FINAL SUMMARY ===')
    print(f'Total evaluations: {results["total_evaluations"]}')
    print(f'Best architecture: {results["final_best_config_label"]}')
    print(
        'Best full-data accuracy: '
        f'{best["test_accuracy"]:.2f}% '
        f'(baseline {baseline["test_accuracy"]:.2f}%)'
    )


if __name__ == '__main__':
    main(sys.argv)
