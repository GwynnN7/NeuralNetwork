# Neural Network

A C++ neural network built using **Eigen3** for vectorized linear algebra. Designed for modularity and easy experimentation.


## Features

* **Flexible Architecture**: Network structure and hyperparameters are fully configurable via a Grid Search CSV file.
* **Rigorous Evaluation**: Nested K-Fold Cross-Validation for model selection and performance estimation. 
* **Model Selection**: Automatically prioritizes models based on maximum Validation Accuracy, using minimum Validation Loss.
* **Activations Supported**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Linear`.
* **Loss Functions**: `MSE` (Mean Squared Error) and `CCE` (Categorical Cross-Entropy) with automatic selection based on output activation.
* **Architecture Features**:
    * `L2 Weight Decay` ($\lambda$) and `Momentum` ($\alpha$).
    * `Mini-batch` and `Stochastic Gradient Descent`.
    * `Random`, `Lecun`, `Glorot`, and `He` weight initialization methods.
* **Datasets Supported**: `XOR`, `XOR_HOT`, and `MNIST`.
* **Logging & Visualization**: CSV logging per-fold and live plotting of Loss/Accuracy curves using `live_plot.py`.

## Prerequisites & Dependencies

* **C++ Compiler**: GCC or Clang supporting **C++20/23** (requires `<print>`, `std::format`, `std::ranges`).
* **CMake**: Version 3.14 or higher.
* **Eigen3**: Installed on your system.
* **CLI11**: Fetched automatically via CMake.

## Building the Project

### Release Build
To enable SIMD vectorization (`-O3 -march=native`), build using the Release configuration:

```bash
mkdir -p build/Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j4
```

## Configuration: Grid Search CSV

Network parameters are defined in a CSV file passed to the `--params` argument. 

**Format (`artifacts/grid.csv`):**
```csv
id,net_struct,hidden,output,init,batch,eta,lambda,alpha
0,128-64,relu,softmax,he,32,0.01,0.001,0.9
1,64,sigmoid,sigmoid,glorot,16,0.1,1e-4,0.0
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
| `--epochs` | Maximum number of training epochs per fold | `500` |
| `--train_ratio`| Training set split ratio (when $K=1$) | `0.85` |
| `--dataset_ratio`| Subset fraction of dataset to load (for fast prototyping) | `1.0` |
| `--shuffle` | Flag to randomly shuffle the dataset before splitting | `false` |
| `--seed` | Random seed for reproducibility | `42` |

## Quickstart Examples

### 1. Training with Nested Cross-Validation on MNIST
Run a 2x2 nested K-fold cross-validation on 40% of the MNIST dataset, evaluating the models defined in `grid.csv`, and dumping the final weights:

```bash
./build/Release/NeuralNet mnist \
    --name mnist_test \
    --train \
    --dump \
    --params artifacts/grid.csv \
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
    --params artifacts/grid.csv \
    --dataset_ratio 0.4
``` 
*(Note: Omitting `--train` automatically triggers inference mode on `.bin` files found in the target directory)*.

## Logging & Output

During training, metrics are continuously aggregated to CSV files in artifact directory (`artifacts/<name>/`). The files follow a naming convention to separate inner-fold validations from outer-fold evaluations:

* **Inner Folds:** `outer<N>_inner_m<ID>.csv` (e.g., `outer0_inner_m0.csv`)
* **Outer Folds (Best Models):** `outer<N>_m<ID>.csv`

**Live Visualization:**
Monitor training in real-time by targeting the artifact directory with the plotting script:
```bash
python live_plot.py mnist_test
```