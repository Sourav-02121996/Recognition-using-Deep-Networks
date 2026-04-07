"""
 * Project 5: Recognition using Deep Networks
 * Course: CS 5330 - Pattern Recognition and Computer Vision
 *
 * <p>Implements Task 4 by re-creating the MNIST classifier with transformer
 * layers and patch embeddings using the provided NetTransformer template
 * structure.</p>
 *
 * <p>Authors: Joseph Defendre, Sourav Das</p>
 """

import argparse
import json
import os
import sys
import random

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

from train_mnist import MNIST_MEAN, MNIST_STD


class NetConfig:
    """Store the Task 4 transformer hyperparameters and dataset settings."""

    def __init__(self,
                 name='vit_base',
                 variant='default',
                 artifact_prefix='transformer',
                 dataset='mnist',
                 patch_size=4,
                 stride=2,
                 embed_dim=48,
                 depth=4,
                 num_heads=8,
                 mlp_dim=128,
                 dropout=0.1,
                 use_cls_token=False,
                 epochs=15,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 seed=0,
                 optimizer='adamw',
                 device='mps',
                 use_conv_stem=False,
                 stem_channels=24,
                 classifier_hidden_dims=None):
        """Initialize the default Task 4 transformer configuration.

        Args:
            name: Human-readable experiment name.
            variant: Variant identifier used for artifact naming and logging.
            artifact_prefix: Prefix used for saved artifact filenames.
            dataset: Dataset identifier used for logging.
            patch_size: Spatial size of each square patch.
            stride: Stride between neighboring patches.
            embed_dim: Token embedding dimension.
            depth: Number of transformer encoder layers.
            num_heads: Attention heads per encoder layer.
            mlp_dim: Feedforward hidden dimension.
            dropout: Dropout probability inside the transformer.
            use_cls_token: Whether to prepend a CLS token or average tokens.
            epochs: Number of training epochs.
            batch_size: Batch size for the data loaders.
            lr: Optimizer learning rate.
            weight_decay: Optimizer weight decay coefficient.
            seed: Random seed for reproducible runs.
            optimizer: Optimizer name.
            device: Preferred device string.
            use_conv_stem: Whether to use a convolutional stem for patch embedding.
            stem_channels: Intermediate channel count used by the convolutional stem.
            classifier_hidden_dims: Hidden dimensions of the classifier MLP head.

        Returns:
            None.
        """
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        self.name = name
        self.variant = variant
        self.artifact_prefix = artifact_prefix
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device
        self.use_conv_stem = use_conv_stem
        self.stem_channels = stem_channels
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [mlp_dim]
        self.classifier_hidden_dims = classifier_hidden_dims

        self.config_string = (
            "Name,Variant,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,"
            "CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
            f"{self.name},{self.variant},{self.dataset},{self.patch_size},{self.stride},"
            f"{self.embed_dim},{self.depth},{self.num_heads},{self.mlp_dim},"
            f"{self.dropout:.2f},{self.use_cls_token},{self.epochs},"
            f"{self.batch_size},{self.lr:.6f},{self.weight_decay:.6f},"
            f"{self.seed},{self.optimizer},"
        )


class PatchEmbedding(nn.Module):
    """Convert an image tensor into a sequence of learned patch tokens."""

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim,
                 use_conv_stem=False, stem_channels=24):
        """Initialize patch extraction and projection modules.

        Args:
            image_size: Width/height of the square input image.
            patch_size: Spatial size of each square patch.
            stride: Stride between neighboring patches.
            in_channels: Number of input image channels.
            embed_dim: Output embedding size for each patch token.
            use_conv_stem: Whether to replace unfold-plus-linear with a conv stem.
            stem_channels: Intermediate channel count used by the conv stem.

        Returns:
            None.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_conv_stem = use_conv_stem
        self.stem_channels = stem_channels

        if self.use_conv_stem:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(stem_channels, embed_dim, kernel_size=patch_size, stride=stride),
            )
            self.unfold = None
            self.patch_dim = embed_dim
        else:
            self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
            self.patch_dim = in_channels * patch_size * patch_size
            self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self):
        """Compute the number of extracted patches for the configured image.

        Args:
            None.

        Returns:
            The total number of patches produced by the unfold step.
        """
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x):
        """Extract patches and project them into token embeddings.

        Args:
            x: Input tensor with shape ``[batch_size, channels, height, width]``.

        Returns:
            A token tensor with shape ``[batch_size, num_patches, embed_dim]``.
        """
        if self.use_conv_stem:
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x

        x = self.unfold(x)
        x = x.transpose(1, 2)
        return self.proj(x)


def build_classifier_head(config):
    """Construct the transformer classification head for the given config.

    Args:
        config: ``NetConfig`` that specifies the head dimensions.

    Returns:
        A ``nn.Sequential`` classifier head.
    """
    layers = []
    in_features = config.embed_dim

    for hidden_dim in config.classifier_hidden_dims:
        layers.extend([
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ])
        in_features = hidden_dim

    layers.append(nn.Linear(in_features, config.num_classes))
    return nn.Sequential(*layers)


class NetTransformer(nn.Module):
    """Vision Transformer for MNIST classification using patch embeddings."""

    def __init__(self, config):
        """Initialize the transformer encoder and classification head.

        Args:
            config: ``NetConfig`` object describing the model hyperparameters.

        Returns:
            None.
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            use_conv_stem=config.use_conv_stem,
            stem_channels=config.stem_channels,
        )

        num_tokens = self.patch_embed.num_patches
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = build_classifier_head(config)

    def forward(self, x):
        """Run a forward pass through the transformer classifier.

        Args:
            x: Input tensor of shape ``[batch_size, 1, 28, 28]``.

        Returns:
            A tensor of log-probabilities with shape ``[batch_size, 10]``.
        """
        x = self.patch_embed(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.use_cls_token:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def set_seed(seed):
    """Seed the random number generators used by the Task 4 script.

    Args:
        seed: Integer random seed value.

    Returns:
        None.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred_device):
    """Resolve the execution device while honoring the preferred setting.

    Args:
        preferred_device: Preferred device string from the config.

    Returns:
        A ``torch.device`` that is available in the current environment.
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


def build_optimizer(model, config):
    """Create the optimizer requested by the configuration.

    Args:
        model: Transformer model whose parameters should be optimized.
        config: ``NetConfig`` containing optimizer settings.

    Returns:
        The configured PyTorch optimizer instance.
    """
    optimizer_name = config.optimizer.lower()
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9,
                         weight_decay=config.weight_decay)
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=config.lr,
                          weight_decay=config.weight_decay)
    return optim.AdamW(model.parameters(), lr=config.lr,
                       weight_decay=config.weight_decay)


def build_mnist_loaders(config):
    """Create the MNIST training and test data loaders for Task 4.

    Args:
        config: ``NetConfig`` containing batch-size settings.

    Returns:
        A tuple ``(train_loader, test_loader)``.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=1000,
        shuffle=False
    )
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer):
    """Train the transformer model for one epoch.

    Args:
        model: Transformer classifier being trained.
        device: Torch device used for training.
        train_loader: MNIST training data loader.
        optimizer: Optimizer used for parameter updates.

    Returns:
        A tuple ``(average_loss, accuracy)`` for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return running_loss / total, 100.0 * correct / total


def evaluate(model, device, data_loader):
    """Evaluate the transformer model on a held-out split.

    Args:
        model: Transformer classifier being evaluated.
        device: Torch device used for evaluation.
        data_loader: Data loader for the target evaluation split.

    Returns:
        A tuple ``(average_loss, accuracy)`` for the evaluation pass.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    return total_loss / total, 100.0 * correct / total


def save_transformer_summary(model, config, save_path):
    """Save the Task 4 model architecture and config summary to disk.

    Args:
        model: Transformer model whose architecture should be saved.
        config: ``NetConfig`` used to build the model.
        save_path: Output path for the architecture summary text file.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total_params = sum(param.numel() for param in model.parameters())
    with open(save_path, 'w', encoding='utf-8') as output_file:
        output_file.write('Task 4 Transformer Configuration\n')
        output_file.write(f'Name: {config.name}\n')
        output_file.write(f'Variant: {config.variant}\n')
        output_file.write(f'Dataset: {config.dataset}\n')
        output_file.write(f'Patch size: {config.patch_size}\n')
        output_file.write(f'Stride: {config.stride}\n')
        output_file.write(f'Embedding dimension: {config.embed_dim}\n')
        output_file.write(f'Depth: {config.depth}\n')
        output_file.write(f'Heads: {config.num_heads}\n')
        output_file.write(f'MLP dimension: {config.mlp_dim}\n')
        output_file.write(f'Dropout: {config.dropout}\n')
        output_file.write(f'Use CLS token: {config.use_cls_token}\n')
        output_file.write(f'Use convolutional stem: {config.use_conv_stem}\n')
        output_file.write(f'Stem channels: {config.stem_channels}\n')
        output_file.write(f'Classifier hidden dims: {config.classifier_hidden_dims}\n')
        output_file.write(f'Total parameters: {total_params}\n\n')
        output_file.write('Model:\n')
        output_file.write(f'{model}\n')


def save_history_plot(history, save_path):
    """Plot the Task 4 training/test curves and save the figure to disk.

    Args:
        history: Dictionary containing loss and accuracy series by epoch.
        save_path: Output image path for the plot.

    Returns:
        None.
    """
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_losses'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['test_losses'], 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer: Training and Test Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, history['train_accuracies'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs, history['test_accuracies'], 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Transformer: Training and Test Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_default_config():
    """Create the baseline Task 4 transformer configuration.

    Returns:
        A ``NetConfig`` matching the original Task 4 settings.
    """
    return NetConfig()


def build_extension_config():
    """Create the transformer extension configuration.

    Returns:
        A ``NetConfig`` using a convolutional patch stem and CLS token head.
    """
    return NetConfig(
        name='vit_extension',
        variant='extension',
        artifact_prefix='transformer_extension',
        patch_size=7,
        stride=4,
        embed_dim=64,
        depth=4,
        num_heads=8,
        mlp_dim=160,
        dropout=0.1,
        use_cls_token=True,
        epochs=10,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        device='cpu',
        use_conv_stem=True,
        stem_channels=24,
        classifier_hidden_dims=[160, 80],
    )


def parse_args(argv):
    """Parse command-line arguments for the transformer script.

    Args:
        argv: Raw command-line argument list.

    Returns:
        Parsed ``argparse.Namespace`` values.
    """
    parser = argparse.ArgumentParser(
        description='Train the baseline or extension transformer on MNIST.'
    )
    parser.add_argument(
        '--variant',
        choices=['default', 'extension'],
        default='default',
        help='Which transformer variant to train and save.'
    )
    return parser.parse_args(argv[1:])


def build_config_for_variant(variant):
    """Create a transformer configuration for the requested variant.

    Args:
        variant: Variant name provided on the command line.

    Returns:
        A configured ``NetConfig`` instance.
    """
    if variant == 'extension':
        return build_extension_config()
    return build_default_config()


def build_output_paths(config):
    """Resolve saved artifact paths for the requested transformer variant.

    Args:
        config: ``NetConfig`` being trained.

    Returns:
        A dictionary of output file paths.
    """
    prefix = config.artifact_prefix
    return {
        'architecture': f'results/{prefix}_architecture.txt',
        'plot': f'results/{prefix}_training.png',
        'history': f'results/{prefix}_history.json',
        'model': f'results/{prefix}_model.pth',
    }


def main(argv):
    """Run the full Task 4 transformer workflow on MNIST.

    Args:
        argv: Command-line arguments passed to the script.

    Returns:
        None.
    """
    args = parse_args(argv)
    config = build_config_for_variant(args.variant)
    output_paths = build_output_paths(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    os.makedirs('results', exist_ok=True)

    print(f'Using device: {device}')
    train_loader, test_loader = build_mnist_loaders(config)

    model = NetTransformer(config).to(device)
    total_params = sum(param.numel() for param in model.parameters())

    print('\nTransformer model:')
    print(model)
    print(f'\nNumber of tokens: {model.patch_embed.num_patches}')
    print(f'Total parameters: {total_params:,}')

    save_transformer_summary(model, config, output_paths['architecture'])
    print(f'Saved transformer architecture to {output_paths["architecture"]}')

    optimizer = build_optimizer(model, config)
    history = {
        'config': {
            'patch_size': config.patch_size,
            'stride': config.stride,
            'embed_dim': config.embed_dim,
            'depth': config.depth,
            'num_heads': config.num_heads,
            'mlp_dim': config.mlp_dim,
            'dropout': config.dropout,
            'use_cls_token': config.use_cls_token,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'weight_decay': config.weight_decay,
            'optimizer': config.optimizer,
            'variant': config.variant,
            'use_conv_stem': config.use_conv_stem,
            'stem_channels': config.stem_channels,
            'classifier_hidden_dims': config.classifier_hidden_dims,
        },
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
    }

    best_test_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, device, train_loader, optimizer
        )
        test_loss, test_accuracy = evaluate(model, device, test_loader)

        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracies'].append(train_accuracy)
        history['test_accuracies'].append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

        print(
            f'Epoch {epoch}: '
            f'Train Loss={train_loss:.4f} Acc={train_accuracy:.2f}%  '
            f'Test Loss={test_loss:.4f} Acc={test_accuracy:.2f}%'
        )

    history['best_test_accuracy'] = best_test_accuracy
    history['best_epoch'] = best_epoch
    history['final_test_accuracy'] = history['test_accuracies'][-1]
    history['final_train_accuracy'] = history['train_accuracies'][-1]
    history['total_parameters'] = total_params
    history['num_tokens'] = model.patch_embed.num_patches

    save_history_plot(history, output_paths['plot'])
    print(f'\nSaved transformer training plot to {output_paths["plot"]}')

    with open(output_paths['history'], 'w', encoding='utf-8') as output_file:
        json.dump(history, output_file, indent=2)
    print(f'Saved transformer history to {output_paths["history"]}')

    torch.save(model.state_dict(), output_paths['model'])
    print(f'Saved transformer model to {output_paths["model"]}')
    print(
        f'{config.variant.capitalize()} variant best test accuracy: {best_test_accuracy:.2f}% '
        f'at epoch {best_epoch}'
    )


if __name__ == '__main__':
    main(sys.argv)
