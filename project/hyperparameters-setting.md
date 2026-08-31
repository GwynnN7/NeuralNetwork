# Hyperparameters Setting — ML-CUP 2025

## 1. Hyperparameters

| Name | Grid column | Meaning | ML-CUP |
|---|---|---|---|
| `net` | 2 | Hidden layer sizes | 1–6 layers, 12–384 units |
| `hidden` | 3 | Hidden activation | `tanh`, `relu`, `sigmoid` |
| `output` | 4 | Output activation | `linear` (regression) |
| `init` | 5 | Weight initialization | `glorot`, `he`, `lecun` |
| `opt` | 6 | Optimizer | `sgd`, `rmsprop`, `adam` |
| `loss` | 7 | Training loss | `mse`, `mee` |
| `norm` | 8 | Data normalization | `minmax`, `zscore`, `absmax` |
| `batch` | 9 | Batch size, 0 = full batch | 0, 16, 32, 64, 128 |
| `eta` | 10 | Initial learning rate | 5e-4 - 1e-2 |
| `lambda` | 11 | L2 regularization | 5e-4 - 1e-1 |
| `dropout` | 12 | Dropout probability | 0, 0.05, 0.1 |
| `alpha` | 13 | SGD momentum / RMSProp decay / Adam β1 | 0.0, 0.85, 0.9, 0.95 |
| `beta` | 14 | Adam β2 (optional, defaults to 0.999) | 0.99, 0.999 |

---

## 2. Search Phases

| Phase | Trials | Inner CV | Outer CV  |
|---|---:|---|---|
| Screening | 2 | 4-fold | Holdout 80/20 | |
| Tuning | 5 | 5-fold | Holdout 80/20 |
| Refinement | 15 | 5-fold | 5-fold |
| Assessment | 15 | 5-fold | 5-fold |


### Ranges Explored

| | Screening | Tuning | Refinement / Assessment |
|---|---|---|---|
| layers | 2–6 | 3–5 | 3–5 |
| units | 12–384 | 12–128 | 32–160 |
| hidden | tanh, relu, sigmoid | tanh, relu | tanh, relu |
| init | glorot, he, lecun | glorot, he, lecun | glorot, he, lecun |
| optimizer | sgd, rmsprop, adam | adam | adam |
| loss | mse, mee | mse | mse |
| normalization | minmax, zscore, absmax | minmax, zscore | minmax, zscore |
| batch | 0, 32, 64, 128 | 0 | 0 |
| eta | 5e-4 - 1e-2 | 1e-3 - 3e-3 | 9e-4 - 2e-3 |
| lambda | 5e-4 - 1e-1 | 1e-3 - 2.5e-3 | 1e-3 - 2.5e-3 |
| dropout | 0, 0.05, 0.1 | 0 | 0 |

---

## 3. Training Settings

| Setting | Value | Note |
|---|---|---|
| `--updates` | 25000 | weight updates, epochs derived from batch size |
| `--warmup` | 50 | updates spent increasing eta from `eta/warmup` to `eta` |
| `--patience` | 1500 | updates without improvement before stopping |
| `--window` | 200 | epochs per window for the selection score |
| decay | linear | eta falls to 1% of the initial value over 80% of the epochs |
| `--stopping` | patience | stop training after `--patience` updates without improvement |
| `--seed` | 42 | mixed with trial, inner-fold and outer-fold indices per run |

---

## 4. Final model

| | |
|---|---|
| Architecture | `96-64-32` (3 hidden layers) |
| Hidden / Output | Tanh / Linear |
| Initialisation | Glorot |
| Optimizer | Adam, β1 0.9, β2 0.999 |
| Normalization | Z-Score |
| Batch | Full |
| eta / lambda | 0.002 / 0.0025 |
| Dropout | 0 |
| Train MEE | 11.59 ± 0.35 |
| Validation MEE | 17.47 ± 0.31 |
| Internal Test MEE | 16.77 ± 0.69 |
| Epoch time | ~0.36 ms |
| Retrain time | ~2.7 s per run |

Reproduce with:

```
# train
./build/Release/NeuralNet mlcup --name mlcup --grid grids/mlcup/shallow.csv \
    --train --shuffle --seed 42 --trials 15 --inner-k 5 --outer-k 5 \
    --updates 25000 --window 200 --patience 1500 --warmup 50 --stopping patience --dump
# test
./build/Release/NeuralNet mlcup --name mlcup
```