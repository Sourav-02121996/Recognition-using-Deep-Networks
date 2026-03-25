# Joseph Defendre, Sourav Das
# CS 5330 - Project 5: Recognition using Deep Networks
# Task 4: Re-implement the network using transformer layers

# import statements
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Transformer-based network for MNIST digit recognition
class NetTransformer(nn.Module):
    """Vision Transformer for MNIST classification using patch embeddings."""

    def __init__(self, image_size=28, patch_size=7, patch_stride=7,
                 num_classes=10, dim=64, depth=2, heads=4,
                 mlp_dim=128, dropout=0.1):
        super(NetTransformer, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Calculate number of patches
        num_patches = ((image_size - patch_size) // patch_stride + 1) ** 2
        patch_dim = patch_size * patch_size  # 1 channel

        # Patch embedding: linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_dim, dim)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Layer norm
        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, num_classes)

    # computes a forward pass through the transformer network
    def forward(self, x):
        batch_size = x.shape[0]

        # Extract patches: x is [B, 1, 28, 28]
        patches = x.unfold(2, self.patch_size, self.patch_stride)\
                   .unfold(3, self.patch_size, self.patch_stride)
        # patches shape: [B, 1, num_h, num_w, patch_size, patch_size]
        patches = patches.contiguous().view(batch_size, -1,
                                            self.patch_size * self.patch_size)
        # patches shape: [B, num_patches, patch_dim]

        # Linear projection to embedding dimension
        x = self.patch_embed(patches)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # Use CLS token output for classification
        x = x[:, 0]

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


# trains the transformer model for one epoch
def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train transformer for one epoch."""
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

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# evaluates the transformer model on the test set
def test_epoch(model, device, test_loader):
    """Evaluate transformer on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# main function
def main(argv):
    """Build, train, and evaluate a transformer-based MNIST classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(42)

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=1000, shuffle=False)

    # Build transformer model with default settings
    model = NetTransformer(
        image_size=28, patch_size=7, patch_stride=7,
        num_classes=10, dim=64, depth=2, heads=4,
        mlp_dim=128, dropout=0.1
    ).to(device)

    print('Transformer model:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader,
                                            optimizer, epoch)
        test_loss, test_acc = test_epoch(model, device, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}%  '
              f'Test Loss={test_loss:.4f} Acc={test_acc:.1f}%')

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), train_losses, 'b-o', label='Train Loss')
    ax1.plot(range(1, epochs + 1), test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer: Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), train_accs, 'b-o', label='Train Acc')
    ax2.plot(range(1, epochs + 1), test_accs, 'r-o', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Transformer: Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('results/transformer_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved transformer training plot to results/transformer_training.png')

    # Save model
    torch.save(model.state_dict(), 'results/transformer_model.pth')
    print('Saved transformer model to results/transformer_model.pth')


if __name__ == "__main__":
    main(sys.argv)
