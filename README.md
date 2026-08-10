# Neural Network

A Neural Network Framework built in **C++** using **Eigen3**. Project designed to be modular and easy to experiment and learn with.

## Features

* **Dynamic Architecture**: Network structure and hyperparameters configurable via a `Grid Search` CSV file.
* **Model Selection**: `Nested K-Fold` Cross-Validation (`Standard` and `Stratified`) or `Holdout`, selecting on error rate for classification and on `MEE` for regression, with `MSE` as tie-break for both.
* **Activations Functions**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Linear`.
* **Loss Functions**: `MSE` (Mean Squared Error), `MEE` (Mean Euclidean Error), `BCE` (Binary Cross-Entropy) and `CCE` (Categorical Cross-Entropy).
* **Architecture Features**:
    * `L2 Weight Decay` ($\lambda$) and `Momentum` ($\alpha$).
    * `Batch`, `Mini-batch` and `Stochastic` update.
    * `SGD`, `RMSProp` and `Adam` optimizers.
    * `Random`, `Lecun`, `Glorot`, and `He` weight initialization methods.
    * `Linear Decay` of `Learning Rate`.
    * `Early Stopping` with `Patience`, `Error` level, or `None`.
* **Datasets Supported**: `XOR(_HOT)`, `MNIST`, `MONK1(_HOT)`, `MONK2(_HOT)`, `MONK3(_HOT)`, `MLCUP`.
    * The `_HOT` datasets are one-hot encoded versions of the original datasets.
* **Logging & Visualization**: 
    * `CSV` logging of every metric for each run.
    * `CLI` interface for training, testing, and visualizing results.
    * `live_plot.py` script to visualize Loss and Accuracy curves.
    * `best_model.py` script to visualize the results of the best model of each fold.

## Configuration

Network parameters are defined in a CSV file passed to the `--grid` argument. 

**Format (`grids/grid.csv`):**

| Item | Description | Options |
| :--- | :--- | :--- |
| `id` | Identifier of the parameters combination (must be unique) | `int` |
| `net` | Structure of the network (`layers` & `neurons`) | `{i}-{i}-{i}` |
| `hidden` | Activation function of the `hidden` layers | `ReLU`, `Sigmoid`, `Tanh`, `Linear` |
| `output` | Activation function of the `output` layer  | `Sigmoid`, `Softmax`, `Linear` |
| `init` | Initialization method of the weights| `Random`, `Lecun`, `Glorot`, `He` |
| `opt` | Optimization method for the weights | `SGD`, `RMSProp`, `Adam` |
| `loss` | Loss type for gradient calcuation | `MSE`, `MEE`, `BCE`, `CCE` |
| `batch` | Batch size (use 0 for a single batch) | `int` &ge; 0 |
| `eta` | Learning rate of the network | `double` > 0 |
| `lambda`| Regularization hyperparameter | `double` &ge; 0 |
| `alpha`| Momentum hyperparameter (`SGD`, `RMSProp`) | `double` in [0, 1) |
| `beta1`| *Optional.* Adam first-moment decay | `double` in [0, 1), default `0.9` |

```csv
id,net,hidden,output,init,opt,loss,batch,eta,lambda,alpha
0,128-64,relu,softmax,he,sgd,cce,32,0.01,0.001,0.9
1,64,sigmoid,sigmoid,glorot,sgd,bce,16,0.1,1e-4,0.0
```

`beta1` is a separate column and optional. If not provided, it defaults to `0.9`. It is only used when the optimizer is `Adam`.

```csv
id,net,hidden,output,init,opt,loss,batch,eta,lambda,alpha,beta1
0,128-64,relu,softmax,he,adam,cce,128,0.001,0.0,0.0,0.9
```

Blank lines and lines beginning with `#` are ignored, so a grid can document what each
block of models is testing — and a single model can be commented out without deleting it:

```csv
# --- Optimiser comparison, each at its own native learning rate ---
0,4,tanh,softmax,glorot,sgd,cce,0,0.5,0,0.85,0.9
#1,4,tanh,softmax,glorot,rmsprop,cce,0,0.01,0,0.9,0.9   <- skipped
2,4,tanh,softmax,glorot,adam,cce,0,0.01,0,0.0,0.9
```

### Hyperparameters correlation

**`alpha` means two different things, while Adam uses `beta1`:**

| Optimizer | `alpha` | `beta1` | Sensible range |
| :--- | :--- | :--- | :--- |
| `SGD` | momentum coefficient | unused | `0.8`–`0.95` (`0` = plain SGD) |
| `RMSProp` | decay of the squared-gradient average | unused | `0.9`–`0.99` |
| `Adam` | **unused** | first-moment decay | `0.9`–`0.99` |

**`eta` is not comparable across optimizers.** `SGD` needs a larger rate than adaptive methods. Note that `eta` is the *initial* rate: it decays linearly to 1% of it over 80% of the epochs that remain after warmup.

**`init` should match `hidden`.** `He` is derived for `ReLU`; `Glorot` and `LeCun` are derived for symmetric saturating
activations (`Tanh`, `Sigmoid`). `Random` is `U(-1,1)` *regardless of fan-in*.

**`lambda` is decoupled from `eta`** and applied directly to the weights, and each mini-batch applies only its own fraction of it.

**`activation` + `loss` constraints:** `Softmax` output requires `CCE` loss and vice versa; `BCE` requires a `Sigmoid` output; regression tasks require `MSE` or `MEE`; `Softmax` is not valid as a hidden activation.

**`output` + `loss` follows the label encoding.** `Softmax`+`CCE` models the outputs as one distribution and works for one-hot targets; `Sigmoid`+`BCE` treats them as independent, non mutually exclusive labels.

### Effective values

| Quantity | Effective value | Notes |
| :--- | :--- | :--- |
| SGD step | $\eta / (1-\alpha)$ | Effective step size increases with high $\alpha$ |
| RMSProp step | $\eta / \sqrt{1-\alpha^{t}} \rightarrow \eta$ | No bias correction, so at low $t$ effective step is increased with high $\alpha$ |
| Adam step | $\approx \eta$ | Bias correction cancels the $\beta_1$ amplification |
| Decay per epoch | $(1-\lambda/B)^{B} \approx 1-\lambda$ | Independent of $B$ = batches per epoch, since each batch scales $\lambda$ by its share of the data |

* **Momentum is not independent.** Changing `alpha` at fixed `eta` changes the step size. To isolate momentum, set `eta` $= s\,(1-\alpha)$ for a fixed target step $s$.
* **`lambda` does not scale with `eta`.** The decay is $w \leftarrow w\,(1-\lambda\,mb/l)$ applied directly to the weights, so changing the learning rate leaves the regularization strength alone.

### Early stopping

The rule is chosen with `--stopping`, with fallbacks when the rule is not applicable:

| Stage | `none` | `patience` | `error` |
| :--- | :--- | :--- | :--- |
| Inner folds  | full epochs | no improvement on **validation** error | *no level yet* → falls back to `patience` |
| Outer retrain | full epochs | no improvement on **training** error | training error reaches the inner folds' validated **training** level |
| Final retrain (`--dump`) | full epochs | no improvement on **training** error | training error reaches the outer retrain's **training** level |

### Model selection metric

Selection never compares training losses: they are not on the same scale across loss functions. Classification is ranked on **error rate** (1 − accuracy, the task objective) and regression on **`MEE`**, both with **`MSE`** as tie-break.


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
| `--epochs` | Maximum number of training epochs per fold | `800` |
| `--stopping` | Early stopping rule: `none`, `error`, `patience` | `patience` |
| `--patience` | Fraction of epochs to wait for improvement before early stopping | `0.125` |
| `--warmup` | Fraction of epochs to warmup the learning rate | `0.1` |
| `--normalization` | Type of normalization to apply to the dataset: `none`, `minmax`, `max`, `zscore` | `none` |
| `--train_ratio`| Training set split ratio, exclusive bounds (when $K=1$) | `0.85` |
| `--dataset_ratio`| Subset fraction of dataset to load (for fast prototyping) | `1.0` |
| `--shuffle` | Flag to randomly shuffle the dataset before splitting | `false` |
| `--seed` | Random seed for reproducibility | `42` |

## Prerequisites & Dependencies

* **C++ Compiler**: GCC or Clang supporting **C++23** (`std::print`, `std::byteswap`).
* **CMake**: Version 3.18 or higher.
* **Eigen3**: Installed on your system.
* **CLI11**: Fetched automatically via CMake.
* **BLAS**: Installed on your system. Optional — disable with `-DNN_USE_BLAS=OFF`.

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
    --grid grids/grid.csv \
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
python scripts/live_plot.py mnist_test
```
Or visualize the performance of the best models found by the model selection foreach outer fold computed
```bash
python scripts/best_model.py mnist_test
```

## Notes
* **`xor`/`xor_hot` replicate 4 distinct points 100 times**, so identical samples are in both train
  and test. The "test" metrics is a convergence check, not a generalization measure.
* **K-fold on `monk_*`/`mnist` shuffles across the datasets' official train/test splits.**
  The predefined split is only used for the holdout case ($K=1$).