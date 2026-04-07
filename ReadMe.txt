================================================================================
                    CS 5330 - Pattern Recognition and Computer Vision
                    Project 5: Recognition using Deep Networks
================================================================================

Name: Joseph Defendre, Sourav Das
Date: April 6, 2026
Course: CS 5330 Pattern Recognition and Computer Vision
Project: Project 5: Recognition using Deep Networks

================================================================================
DESCRIPTION
================================================================================
This project implements several deep-learning pipelines for handwritten-image
recognition using PyTorch. The work includes:

1. A CNN trained on MNIST for handwritten digit recognition.
2. Evaluation of the trained CNN on the MNIST test set and on custom
   handwritten digit images.
3. Visualization and analysis of the first convolutional-layer filters.
4. Transfer learning from the MNIST CNN to classify Greek letters
   (alpha, beta, gamma).
5. A transformer-based MNIST classifier.
6. A transformer extension with a convolutional patch-embedding stem, CLS token,
   and deeper classifier head.
7. An automated experiment on Fashion MNIST exploring three CNN architecture
   dimensions.

The final report is provided in both HTML and PDF form. The PDF intended for
submission is `report.pdf`.


================================================================================
QUICK START
================================================================================
The easiest way to verify this submission on another system is:

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Run these quick verification commands:

   python3 task1_part_d.py
   python3 task1_part_e.py
   python3 task2_part_a.py
   python3 task2_part_b.py

These are the fastest scripts to confirm that the saved MNIST model and the
main Task 1 / Task 2 outputs are present and working.

5. To inspect the submitted write-up and generated figures directly:

   report.pdf
   results/

Important included data:
- The repository already includes the MNIST raw files in `data/MNIST/`.
- The repository already includes the FashionMNIST raw files in
  `data/FashionMNIST/`.
- The Greek training set is already present in `greek_train/`.
- The custom handwritten digit images are already present in `custom_digits/`.
- The custom Greek images are already present in `custom_greek/`.

Because these files are included, no additional data folders or handwritten
examples are required to run the submitted code.


================================================================================
BUILDING INSTRUCTIONS
================================================================================
1. Create and activate a Python virtual environment:

   python3 -m venv venv
   source venv/bin/activate

2. Install the required packages:

   pip install -r requirements.txt

   If `pip` is not available on the system, use:

   pip3 install -r requirements.txt

3. Run the task scripts as needed. Common entry points:

   python3 train_mnist.py
   python3 test_mnist.py
   python3 examine_network.py
   python3 transfer_learning.py
   python3 net_transformer.py
   python3 net_transformer.py --variant extension
   python3 experiment.py

4. Standalone part-by-part entry points are also included:

   python3 task1_part_a.py
   python3 task1_part_b.py
   python3 task1_part_c.py
   python3 task1_part_d.py
   python3 task1_part_e.py
   python3 task1_part_f.py
   python3 task2_part_a.py
   python3 task2_part_b.py

5. All commands should be run from the project root directory.

================================================================================
DEVELOPMENT ENVIRONMENT
================================================================================
Operating System:
- Developed in a local macOS environment.

Language and Runtime:
- Python 3

Primary Libraries:
- PyTorch
- torchvision
- matplotlib
- numpy
- Pillow
- OpenCV

Configuration Notes:
- Matplotlib output is configured to use a local `.matplotlib/` directory.
- Scripts automatically create and use the `results/` directory for generated
  figures, saved models, and metrics.
- The code falls back to CPU if CUDA or Apple MPS acceleration is not
  available.


================================================================================
FILE STRUCTURE
================================================================================
Root-level source files:
- train_mnist.py
- test_mnist.py
- examine_network.py
- transfer_learning.py
- net_transformer.py
- experiment.py
- task1_part_a.py
- task1_part_b.py
- task1_part_c.py
- task1_part_d.py
- task1_part_e.py
- task1_part_f.py
- task2_part_a.py
- task2_part_b.py

Documentation and metadata:
- report.pdf
- report.html
- README.md
- ReadMe.txt
- requirements.txt

Data and custom-input folders:
- data/
- greek_train/
- custom_digits/
- custom_greek/

Generated output folder:
- results/

Assignment reference files:
- Project 5_ Recognition using Deep Networks.html
- Project 5_ Recognition using Deep Networks_files/
- NetTransformer-template.py

================================================================================
REQUIRED FILES
================================================================================
The following files are important for building, running, and submitting the
project:

Submission documents:
- report.pdf
- ReadMe.txt

Project source:
- train_mnist.py
- test_mnist.py
- examine_network.py
- transfer_learning.py
- net_transformer.py
- experiment.py

Task wrappers:
- task1_part_a.py
- task1_part_b.py
- task1_part_c.py
- task1_part_d.py
- task1_part_e.py
- task1_part_f.py
- task2_part_a.py
- task2_part_b.py

Environment:
- requirements.txt

Generated artifacts commonly referenced in the report:
- results/training_plot.png
- results/first_nine_predictions.png
- results/custom_digits_results.png
- results/conv1_filters.png
- results/filter_effects.png
- results/greek_training_loss.png
- results/custom_greek_results.png
- results/transformer_training.png
- results/transformer_extension_training.png
- results/experiment_results.png

================================================================================
RUNNING NOTES
================================================================================
1. MNIST and Fashion MNIST data are downloaded automatically by torchvision the
   first time the scripts are run.

2. `transfer_learning.py` expects the Greek training set in either:
   - greek_train/
   - data/greek_train/
   with `alpha/`, `beta/`, and `gamma/` subfolders.

3. `task1_part_f.py` and `test_mnist.py` expect custom handwritten digit images
   in `custom_digits/` using filenames 0 through 9 with common image
   extensions such as PNG, JPG, JPEG, or BMP.

4. `transfer_learning.py` optionally evaluates custom Greek-letter images from
   `custom_greek/`.

5. The transformer extension can be run separately with:
   python3 net_transformer.py --variant extension

6. The report references generated files in `results/`, so those artifacts
   should be regenerated if the models are retrained.

7. Since `data/MNIST/` and `data/FashionMNIST/` are already included in this
   submission, internet access should not be required for the provided data to
   run on another system.

8. If `custom_digits/` or `custom_greek/` are missing, the corresponding custom
   evaluation scripts will skip those inputs or print a clear message; the main
   MNIST and filter-analysis tasks will still run.

9. `transfer_learning.py` requires the included `greek_train/` directory.

10. `experiment.py` is the slowest script in the repository because it trains
    many CNN variants for the architecture-search task.

11. `net_transformer.py --variant extension` is also slower than the quick
    verification scripts because it trains an extended transformer from scratch.


================================================================================
FILE DESCRIPTION
================================================================================
- `task1_part_d.py` checks that the saved MNIST model file exists and loads.
- `task1_part_e.py` prints the first 10 MNIST test predictions and saves the
  3x3 prediction figure.
- `task2_part_a.py` prints the model structure and conv1 filter weights and
  saves the first-layer filter visualization.
- `task2_part_b.py` applies the learned conv1 filters to the first training
  image and saves the filter-effect figure.
- `transfer_learning.py` trains a 3-class Greek-letter classifier using the
  frozen MNIST feature extractor and saves the training history and outputs.
- `net_transformer.py` trains the baseline transformer for Task 4.
- `net_transformer.py --variant extension` trains the added transformer
  extension with a convolutional patch stem, CLS token, and deeper classifier.
- `experiment.py` runs the automated Fashion MNIST experiment and saves raw
  metrics plus the summary plot used in the report.


================================================================================
OUTPUTS GENERATED
================================================================================
Important output files produced by the scripts include:

MNIST CNN:
- results/mnist_model.pth
- results/training_history.json
- results/training_plot.png
- results/test_set_outputs.txt
- results/first_nine_predictions.png
- results/custom_digit_outputs.txt
- results/custom_digits_inputs.png
- results/custom_digits_results.png

Network analysis:
- results/conv1_weights.txt
- results/conv1_filters.png
- results/filter_effects.png

Greek transfer learning:
- results/greek_model.pth
- results/greek_model_architecture.txt
- results/greek_training_history.json
- results/greek_training_loss.png
- results/custom_greek_outputs.txt
- results/custom_greek_results.png

Transformer:
- results/transformer_model.pth
- results/transformer_architecture.txt
- results/transformer_history.json
- results/transformer_training.png

Transformer extension:
- results/transformer_extension_model.pth
- results/transformer_extension_architecture.txt
- results/transformer_extension_history.json
- results/transformer_extension_training.png

Experiment:
- results/experiment_results.json
- results/experiment_results.png
- results/experiment_summary.txt

================================================================================
ACKNOWLEDGEMENTS
================================================================================
- Claude (Anthropic)
  Used as a generative AI assistant for help with code debugging, and report writing.
- PyTorch tutorials and official documentation were consulted for model design,
  training loops, and tensor operations.
- The Greek letter training images were provided as course materials.
- The MNIST and Fashion MNIST datasets are provided through torchvision.

================================================================================
TIME TRAVEL DAYS
================================================================================
Time travel days used: 0

================================================================================
ADDITIONAL DETAILS
================================================================================
- No videos are included with this submission.
- Custom Greek examples link:
  https://drive.google.com/drive/folders/1qOwOHZjCKCoBXG1X0rVXl0GSploeAE9X?usp=sharing
- The transformer extension added for the project reached 98.95% test accuracy
  after 10 epochs and is documented in the report.
