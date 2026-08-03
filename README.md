# Neural Network

A Neural Network Framework built in **C++** using **Eigen3**. Project designed to be modular and easy to experiment and learn with.

## Features

* **Dynamic Architecture**: Network structure and hyperparameters configurable via a `Grid Search` CSV file.
* **Model Selection**: `Nested K-Fold` Cross-Validation or `Holdout` to select the best model based on Loss.
* **Activations Functions**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Linear`.
* **Loss Functions**: `MSE` (Mean Squared Error), `BCE` (Binary Cross-Entropy) and `CCE` (Categorical Cross-Entropy).
* **Architecture Features**:
    * `L2 Weight Decay` ($\lambda$) and `Momentum` ($\alpha$).
    * `Batch`, `Mini-batch` and `Stochastic` update.
    * `SGD`, `RMSProp` and `Adam` optimizers.
    * `Random`, `Lecun`, `Glorot`, and `He` weight initialization methods.
    * `Linear Decay` of `Learning Rate`.
    * `Early Stopping` with `Patience`.
* **Datasets Supported**: `XOR`, `XOR_HOT`, and `MNIST`.
* **Logging & Visualization**: 
    * `CSV` logging of per-fold loss and accuracy.
    * `live_plot.py` script to visualize Loss and Accuracy curves.
    * `best_model.py` script to visualize the results of the best model of each fold.

## Configuration

Network parameters are defined in a CSV file passed to the `--params` argument. 

**Format (`dataset/<dataset>/grid.csv`):**

| Item | Description | Options |
| :--- | :--- | :--- |
| `id` | Identifier of the parameters combination | `int` |
| `net` | Structure of the network (`layers` & `neurons`) | `{i}-{i}-{i}` |
| `hidden` | Activation function of the `hidden` layers | `ReLU`, `Sigmoid`, `Tanh` |
| `output` | Activation function of the `output` layer  | `Sigmoid`, `Softmax`, `Linear` |
| `init` | Initialization method of the weights| `Random`, `Lecun`, `Glorot`, `He` |
| `opt` | Optimization method for the weights | `SGD`, `RMSProp`, `Adam` |
| `loss` | Loss type for gradient calcuation | `MSE`, `BCE`, `CCE` |
| `batch` | Batch size (use 0 for a single batch) | `int` |
| `eta` | Learning rate of the network | `double` |
| `lambda`| Regularization hyperparameter | `double` |
| `alpha`| Momentum hyperparameter | `double` |

```csv
id,net,hidden,output,init,opt,batch,eta,lambda,alpha
0,128-64,relu,softmax,he,sgd,32,0.01,0.001,0.9
1,64,sigmoid,sigmoid,glorot,sgd,16,0.1,1e-4,0.0
```


## CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `dataset` | Dataset type: `xor`, `xor_hot`, `mnist` | *Required* |
| `--params` | Path to CSV file containing model configurations for grid search | `dataset/grid.csv` |
| `--name` | Name for the project run (creates `artifacts/<name>/` directory) | `model` |
| `--train` | Flag to execute the training & cross-validation loop | `false` |
| `--dump` | Flag to serialize and save the best trained models to `.bin` files | `false` |
| `--inner-k` | Number of folds for inner cross-validation (Model Selection) | `1` |
| `--outer-k` | Number of folds for outer cross-validation (Model Evaluation) | `1` |
| `--epochs` | Maximum number of training epochs per fold | `800` |
| `--patience` | Number of epochs to wait before checking for early stopping | `60` |
| `--train_ratio`| Training set split ratio (when $K=1$) | `0.85` |
| `--dataset_ratio`| Subset fraction of dataset to load (for fast prototyping) | `1.0` |
| `--shuffle` | Flag to randomly shuffle the dataset before splitting | `false` |
| `--seed` | Random seed for reproducibility | `42` |

## Prerequisites & Dependencies

* **C++ Compiler**: GCC or Clang supporting **C++20/23**.
* **CMake**: Version 3.14 or higher.
* **Eigen3**: Installed on your system.
* **CLI11**: Fetched automatically via CMake.
* **BLAS**: Installed on your system [can be disabled via CMake]

## Building the Project

### Release Build
For optimal performance, build using the Release configuration:

```bash
mkdir -p build/Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j4
```

## Quickstart Examples

### Use provided `train` and `test` scripts for enhanced features
```bash
# Train the models of <grid_file> on <dataset>. 
./train <dataset> <name> <grid_file>

# Test the best models <name> on <dataset>. 
./test <dataset> <name>
```
This provides a menu for triggering early stopping, quitting or visualizing plots. The `name` and `grid_file` arguments are optional. More configurations available in the bash script.

### 1. Training with Nested Cross-Validation on MNIST
Run a 2x2 nested K-fold cross-validation on 40% of the MNIST dataset, evaluating the models defined in `grid.csv`, and dumping the final weights:

```bash
./build/Release/NeuralNet mnist \
    --name mnist_test \
    --train \
    --dump \
    --params dataset/mnist/grid.csv \
    --dataset_ratio 0.4 \
    --inner-k 2 \
    --outer-k 2 \
    --epochs 200 \
    --shuffle
```

### 2. Loading and Inference
To load previously dumped models from the `artifacts/mnist_test` directory and run inference against the dataset:

```bash
./build/Release/NeuralNet mnist \
    --name mnist_test \
    --dataset_ratio 0.4
``` 
*(Note: Omitting `--train` triggers inference mode on `.bin` files found in the target directory)*.

## Logging & Output

During training, metrics are logged to CSV files in artifact directory (`artifacts/<name>/`). The files follow a naming convention to separate inner-fold validations from outer-fold evaluations:

* **Inner Folds:** `outer<N>_inner_m<ID>.csv` (e.g., `outer0_inner_m0.csv`)
* **Outer Folds:** `outer<N>_m<ID>.csv` (e.g., `outer1_m1.csv`) (Best Models)

**Visualization:**
Monitor training in real-time by targeting the artifact directory with the plotting script:
```bash
python live_plot.py mnist_test
```
Or visualize the performance of the best models found by the model selection foreach outer fold computed
```bash
python best_model.py mnist_test
```