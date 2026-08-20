# Neural Network

A Neural Network Framework built in **C++** using **Eigen3**. Project designed to be modular and easy to experiment and learn with.

## Features

* **Dynamic Architecture**: Network structure and hyperparameters configurable via a `Grid Search` CSV file.
* **Model Selection**: `Nested K-Fold` Cross-Validation (`Standard` and `Stratified`) or `Holdout`, selecting on error rate for classification and on `MEE` for regression, with `MSE` as tie-break for both.
* **Activations Functions**: `ReLU`, `Leaky ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Linear`.
* **Loss Functions**: `MSE` (Mean Squared Error), `MEE` (Mean Euclidean Error), `BCE` (Binary Cross-Entropy) and `CCE` (Categorical Cross-Entropy).
* **Architecture Features**:
    * `L2 Weight Decay` ($\lambda$) and `Momentum` ($\alpha$).
    * `Batch`, `Mini-batch` and `Stochastic` update.
    * `SGD`, `RMSProp` and `Adam` optimizers.
    * `Random`, `Lecun`, `Glorot`, and `He` weight initialization methods.
    * `Warmup` and `Linear Decay` of `Learning Rate`.
    * `Early Stopping` with `Patience`, `Error` level, or `None`.
* **Datasets Supported**: `XOR(_HOT)`, `MNIST`, `MONK1(_HOT)`, `MONK2(_HOT)`, `MONK3(_HOT)`, `MLCUP`.
    * The `_HOT` datasets are one-hot encoded versions of the original datasets.
* **Logging**: 
    * `CSV` logging of every metric for each run.
* **Visualization [optional, not part of the project so no explicit requirements declared]**: 
    * `CLI` interface for training, testing, and visualizing results.
    * `live_plot.py` script to visualize Loss and Accuracy curves.
    * `best_model.py` script to visualize the results of the best model of each fold.

## Prerequisites & Dependencies

* **C++ Compiler**: GCC or Clang supporting **C++23**.
* **CMake**: Version 3.18 or higher.
* **Eigen3**: Must be installed system-wide. More information [here](https://libeigen.gitlab.io/).
* **OpenBLAS** *(optional, disable with `-DNN_USE_BLAS=OFF`, default on)*: Must be installed system-wide. More information [here](https://www.openmathlib.org/OpenBLAS/).
* **OpenMP**: Must be installed system-wide. used by OpenBLAS.
* **CLI11**: Fetched automatically, downloaded at configure time by CMake. More information [here](https://cliutils.github.io/CLI11/).

```bash
# Debian / Ubuntu
sudo apt install libeigen3-dev libopenblas-dev libomp-dev
# Arch
sudo pacman -S eigen openblas openmp
```

## Building the Project

### Release Build
For optimal performance, build using the Release configuration:

```bash
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release && cmake --build build/Release -j4
```

### Build options

| Option | Description | Default |
| :--- | :--- | :--- |
| `NN_USE_BLAS` | Use an external BLAS library for matrix operations | `ON` |
| `NN_NATIVE_ARCH` | Build with `-march=native` (non-portable binary) | `ON` |
| `NN_DOUBLE_PRECISION` | Use `double` instead of `float` for `Scalar` | `OFF` |
| `NN_SANITIZE` | Build with ASan + UBSan | `OFF` |

## Configuration

Network parameters are defined in a CSV file passed to the `--grid` argument. 

**Format (`grids/grid.csv`):**

| Item | Description | Options |
| :--- | :--- | :--- |
| `id` | Identifier of the parameters combination (must be unique) | `int` |
| `net` | Structure of the network (`layers` & `neurons`) | `{i}-{i}-{i}` |
| `hidden` | Activation function of the `hidden` layers | `ReLU`, `Leaky_ReLU`, `Sigmoid`, `Tanh`, `Linear` |
| `output` | Activation function of the `output` layer  | `Sigmoid`, `Softmax`, `Linear` |
| `init` | Initialization method of the weights| `Random`, `Lecun`, `Glorot`, `He` |
| `opt` | Optimization method for the weights | `SGD`, `RMSProp`, `Adam` |
| `loss` | Loss type for gradient calcuation | `MSE`, `MEE`, `BCE`, `CCE` |
| `norm` | Normalization method for the dataset | `none`, `minmax`, `max`, `zscore` |
| `batch` | Batch size (use 0 for a single batch) | `int` &ge; 0 |
| `eta` | Learning rate of the network | `double` > 0 |
| `lambda`| Regularization hyperparameter | `double` &ge; 0 |
| `dropout`| Dropout rate for hidden layers | `double` in [0, 1), default `0.0` |
| `alpha`| First moment: momentum (`SGD`), decay (`RMSProp`), $\beta_1$ (`Adam`) | `double` in [0, 1) |
| `beta`| *Optional.* Second moment: $\beta_2$ (`Adam` only) | `double` in [0, 1), default `0.999` |

```csv
id,net,hidden,output,init,opt,loss,norm,batch,eta,lambda,dropout,alpha
0,128-64,relu,softmax,he,sgd,cce,minmax,32,0.01,0.001,0,0.9
1,64,sigmoid,sigmoid,glorot,sgd,bce,none,16,0.1,1e-4,0,0.0
```

Blank lines and lines beginning with `#` are ignored, and serves as comments.

`beta` is a separate column and optional. If not provided, it defaults to `0.999`. It is only used when the optimizer is `Adam`.

```csv
id,net,hidden,output,init,opt,loss,norm,batch,eta,lambda,dropout,alpha,beta
0,128-64,relu,softmax,he,adam,cce,minmax,128,0.001,0.0,0,0.9,0.999
```


## CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `dataset` | Dataset type: `xor`, `xor_hot`, `MONK1(_hot)`, `MONK2(_hot)`, `MONK3(_hot)`, `mnist`, `MLCUP` | *Required* |
| `--grid` | Path to CSV file containing model configurations for grid search | `grids/grid.csv` |
| `--name` | Name for the project run (creates `artifacts/<name>/` directory) | `model` |
| `--train` | Flag to execute the training & cross-validation loop | `false` |
| `--dump` | Flag to serialize and save the best trained models to `.bin` files | `false` |
| `--inner-k` | Number of folds for inner cross-validation (Model Selection) | `1` |
| `--outer-k` | Number of folds for outer cross-validation (Model Evaluation) | `1` |
| `--trials` | Number of trials for averaging results with different initializations| `1` |
| `--updates` | Number of weight updates per run, converted to epochs per model | `800` |
| `--stopping` | Early stopping rule: `none`, `error`, `patience` | `patience` |
| `--patience` | Weight updates without improvement before early stopping | `75` |
| `--warmup` | Weight updates spent warming the learning rate up | `50` |
| `--window` | Epochs per window for the model selection score | `50` |
| `--train_ratio`| Training set split ratio (when $K=1$) | `0.85` |
| `--dataset_ratio`| Subset fraction of dataset to load | `1.0` |
| `--shuffle` | Flag to randomly shuffle the dataset before splitting | `false` |
| `--seed` | Random seed for reproducibility | `42` |



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
    --grid grids/mnist/mnist.csv \
    --dataset_ratio 0.4 \
    --inner-k 2 \
    --outer-k 2 \
    --updates 20000 \
    --patience 3000 \
    --window 100 \
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
python scripts/live_plot.py mnist_test
```
Or visualize the performance of the best models found by the model selection foreach outer fold computed
```bash
python scripts/best_model.py mnist_test
```

## Notes
### Datasets Notes
* **`xor`/`xor_hot` replicate 4 distinct points 100 times**, so identical samples are in both train
  and test. The "test" metrics is a convergence check, not a generalization measure.
* **K-fold on `monk_*`/`mnist` shuffles across the datasets' official train/test splits.**
  The predefined split is only used for the holdout case (Outer $K=1$).

### Hyperparameters Notes

**`alpha` is the exponential average the optimizer keeps first, so it means a different thing in each. `beta` is the second one, which only `Adam` has:**

| Optimizer | `alpha` | `beta` | Best range for `alpha` |
| :--- | :--- | :--- | :--- |
| `SGD` | momentum coefficient | unused | `0.8`–`0.95` (`0` = standard SGD) |
| `RMSProp` | decay of the squared-gradient average | unused | `0.9`–`0.99` |
| `Adam` | first-moment decay ($\beta_1$) | second-moment decay ($\beta_2$) | `0.9`–`0.99` |

**`lambda` is independent of `eta` and `alpha`** and applied directly to the weights. Each mini-batch applies only its own fraction, $\lambda\,(mb/l)$.

**`activation` + `loss` constraints:** `Softmax` output requires `CCE` loss and vice versa; `BCE` requires a `Sigmoid` output; regression tasks require `MSE` or `MEE`; `Softmax` is not valid as a hidden activation.

### Early stopping

The rule is chosen with `--stopping`, with fallbacks when the rule is not applicable:

`--updates`, `--patience` and `--warmup` are counted in **weight updates**, not epochs. One epoch is a single
update at full batch, but $\lceil l/mb \rceil$ in mini-batch.

| Type | `none` | `patience` | `error` |
| :--- | :--- | :--- | :--- |
| Inner folds  | full epochs | no improvement on **validation** error | falls back to `patience` |
| Outer retrain | full epochs | no improvement on **training** error | training error reaches the inner folds' validated **training** level |
| Final retrain (`--dump`) | full epochs | no improvement on **training** error | training error reaches the outer retrain's **training** level |

### Model selection metric

Classification is ranked on **error rate** (1 − accuracy, the task objective) and Regression on
**`MEE`**, both with **`MSE`** as tie-break.

Each run is scored over a sliding
window of `--window` epochs:

$$\text{window score} = \text{mean}(w) + 0.5 \cdot \text{std}(w), \qquad
  \text{run score} = \min_{w} \text{window score}(w)$$