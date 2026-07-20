# Neural Network

A C++ neural network built using **Eigen3** for vectorized linear algebra. Designed for modularity and easy experimentation.


## Key Features

* **Flexible Architecture**: Network structure and hyperparameters configurable via the command line.
* **Activations Supported**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Linear`.
* **Loss Functions**: `MSE` (Mean Squared Error) and `CCE` (Categorical Cross-Entropy) with automatic selection based on output activation.
* **Architecture Features**:
    * `L2 Weight Decay` ($\lambda$).
    * `Momentum` ($\alpha$)
    * `Batch`, `Mini-batch`, `Stochastic` Gradient Descent.
    * `Random`, `Lecun`, `Glorot`, `He` weights initialization methods.
  
* **Datasets Supported**:
  * `XOR` & `XOR_HOT` (1-neuron binary & multi-class XOR).
  * `MNIST` (Handwritten digit recognition).
* **Logging**: Live plotting of Loss and Accuracy Curves by logging per-epoch loss and accuracy metrics to a CSV file.


## Prerequisites & Dependencies

* **C++ Compiler**: GCC or Clang supporting C++17.
* **CMake**: Version 3.14 or higher.
* **Eigen3**: Installed on your system.
* **CLI11**: Fetched automatically via CMake.


## Building the Project

### Release Build (Recommended for Speed)
To enable SIMD vectorization (`-O3 -march=native`), build using the Release configuration:

```bash
mkdir -p build/Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j4
```

### Debug Build
For debugging with optimizations disabled, build using the Debug configuration:

```bash
mkdir -p build/Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug -j4
```


## CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `dataset` | Dataset type: `xor`, `xor_hot`, `mnist` | *Required* |
| `--network` | List of hidden layer sizes (e.g., `--network 128 64`) | `2 1` |
| `--hidden` | Hidden activation: `sigmoid`, `relu`, `tanh`, `softmax`, `linear` | `sigmoid` |
| `--output` | Output activation: `linear`, `sigmoid`, `relu`, `tanh`, `softmax` | `linear` |
| `--init` | Weight initialization method: `lecun`, `random`, `glorot`, `he` | `lecun` |
| `--epochs` | Number of training epochs | `1000` |
| `--batch_size` | Mini-batch size | `1` |
| `--eta` | Learning rate ($\eta$) | `0.5` |
| `--lambda` | L2 Regularization / Weight decay ($\lambda$) | `0` |
| `--alpha` | Momentum multiplier ($\alpha$) | `0` |
| `--train_ratio` | Training set split ratio | `0.8` |
| `--dataset_ratio` | Subset fraction of dataset to load (for fast testing) | `1.0` |
| `--log` | Output CSV log filename | `log.csv` |
| `--dump` | Output file path to dump model weights | *None* |
| `--load` | Input file path to load model weights | *None* |


## Quickstart Examples
### Launch Script
Use the available scripts to launch the project
```bash
./launch
python plot.py log.csv
```

### 1. Standard XOR Problem (Single Output Neuron)
Train a 2-layer network on the binary XOR problem:

```bash
./$RELEASE_DIR/NeuralNet xor \
    --network 4 \
    --hidden sigmoid \
    --output sigmoid \
    --epochs 200 \
    --eta 0.5
```

### 2. MNIST Classification
Train a 128-64 MLP on MNIST using ReLU and Softmax/CCE loss:

```bash
./$RELEASE_DIR//NeuralNet mnist \
    --network 128 64 \
    --hidden relu \
    --output softmax \
    --init he \
    --epochs 500 \
    --batch 128 \
    --eta 0.4 \
    --dataset_ratio 1.0
```

### 3. Rapid Prototyping (Subset of MNIST)
Train quickly on 10% of the MNIST dataset:

```bash
./$RELEASE_DIR//NeuralNet mnist \
    --network 64 \
    --hidden sigmoid \
    --output sigmoid \
    --epochs 100 \
    --batch 32 \
    --eta 0.1 \
    --dataset_ratio 0.1
```


## Logging & Output

During training, metrics are continuously appended to `build/<log_file>` in CSV format:

```csv
epoch,train_loss,test_loss,train_acc,test_acc
0,2.3025,2.3012,11.2,10.8
1,0.4120,0.3890,88.4,89.1
...
```