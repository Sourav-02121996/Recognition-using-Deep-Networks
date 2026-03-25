# Project 5: Recognition using Deep Networks

**Names:** Joseph Defendre, Sourav Das
**Course:** CS 5330 - Pattern Recognition and Computer Vision
**Date:** March 2026

## Links/URLs
- No videos submitted for this project.

## Time Travel Days
- Time travel days used: 0

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Code

### Task 1: Train MNIST Network
```bash
python3 train_mnist.py
```
This trains the CNN on MNIST digits for 5 epochs and saves the model to `results/mnist_model.pth`.

### Task 1 (continued): Test on MNIST and Custom Digits
```bash
python3 test_mnist.py
```
Place your handwritten digit images (0.png through 9.png) in a `custom_digits/` directory before running.

### Task 2: Examine the Network
```bash
python3 examine_network.py
```
Analyzes conv1 filters and shows their effects on a sample digit.

### Task 3: Transfer Learning on Greek Letters
```bash
python3 transfer_learning.py
```
Download and extract the Greek letters dataset into `greek_train/` with subdirectories `alpha/`, `beta/`, `gamma/` before running.
Place custom Greek letter images in `custom_greek/` to test.

### Task 4: Transformer Network
```bash
python3 net_transformer.py
```
Trains a Vision Transformer on MNIST digits.

### Task 5: Experiment
```bash
python3 experiment.py
```
Evaluates network architecture variations on Fashion MNIST across 3 dimensions.

## File Structure
- `train_mnist.py` — Task 1: Build and train CNN
- `test_mnist.py` — Task 1.5-1.6: Test on MNIST + custom digits
- `examine_network.py` — Task 2: Analyze network filters
- `transfer_learning.py` — Task 3: Greek letter transfer learning
- `net_transformer.py` — Task 4: Transformer-based classifier
- `experiment.py` — Task 5: Architecture experimentation
- `results/` — Output plots and saved models
- `report.html` — Project report (PDF generated from HTML)
