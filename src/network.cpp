#include "network.hpp"

#include "dataset.hpp"
#include "datasplit.hpp"
#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <print>
#include <ranges>
#include <string>
#include <vector>

// Network constructor that initializes model parameters and sets the loss function
Network::Network(const Model& model) : model(model) {
    setLossFunction(model.loss_type);
}

// Network constructor that calls the layers building function with the specified number of features and classes
Network::Network(const Model& model, int num_features, int num_classes) : Network(model) {
    buildLayers(model, num_features, num_classes, nullptr, NetworkMode::TRAIN);
}

// Network constructor that calls the layers building function with the specified weights and biases for each layer
Network::Network(const Model& model, const std::vector<Parameters>& params) : Network(model) {
    if (params.size() != model.net_struct.size() + 1) {
        throw std::invalid_argument("Mismatch between the number of parameter sets and layers in the model");
    }
    buildLayers(model, static_cast<int>(params.front().W.cols()), static_cast<int>(params.back().W.rows()), &params, NetworkMode::TEST);
}

// Function that builds each layer of the network, either loaded or initialized
void Network::buildLayers(const Model& model, int num_features, int num_classes, const std::vector<Parameters>* params, NetworkMode mode) {
    validateModel(model, num_features, num_classes);

    for (size_t i = 0; i < model.net_struct.size(); ++i) {
        // Add a DenseLayer and an ActivationLayer for each layer in the network structure
        if (params != nullptr) {
            addLayer(std::make_unique<DenseLayer>(params->at(i), model.opt_type, mode));
        } else {
            const int input_features = (i == 0) ? num_features : model.net_struct[i - 1];
            addLayer(std::make_unique<DenseLayer>(input_features, model.net_struct[i], model.init_type, model.opt_type));
        }
        addLayer(std::make_unique<ActivationLayer>(model.hidden_activation));
        if (i < model.net_struct.size() - 1 && model.dropout > Scalar(0) && mode == NetworkMode::TRAIN) {
            addLayer(std::make_unique<DropoutLayer>(model.dropout));
        }
    }

    // Add the final output layer with the specified number of classes and output activation
    if (params != nullptr) {
        addLayer(std::make_unique<DenseLayer>(params->back(), model.opt_type, mode));
    } else {
        addLayer(std::make_unique<DenseLayer>(model.net_struct.back(), num_classes, model.init_type, model.opt_type));
    }

    // Add the output activation layer. Only this layer can have its derivative handled by the loss
    const bool derivative_in_loss = LossFunctions::includes_output_derivative(model.loss_type);
    addLayer(std::make_unique<ActivationLayer>(model.output_activation, derivative_in_loss));
}

// Utility function to set the loss function and its derivative based on the specified LossType
void Network::setLossFunction(LossType lossType) {
    const std::optional<LossPair> functions = Lookup::loss_for(lossType);
    if (!functions) {
        throw std::invalid_argument("Unsupported loss function type");
    }
    loss_pair = *functions;
}

// Reject invalid configurations of the network
void Network::validateModel(const Model& model, int num_features, int num_classes) {
    if (model.net_struct.empty()) {
        throw std::invalid_argument("Network structure is empty");
    }
    if (num_features <= 0) {
        throw std::invalid_argument("Dataset has no features");
    }
    if (num_classes <= 0) {
        throw std::invalid_argument("Dataset has no output classes");
    }
    if (model.hidden_activation == ActivationType::SOFTMAX) {
        throw std::invalid_argument("Softmax is not supported as a hidden activation");
    }
    if (model.task == TaskType::REGRESSION && model.loss_type != LossType::MSE && model.loss_type != LossType::MEE) {
        throw std::invalid_argument("Regression tasks must use MSE or MEE loss");
    }
    if ((model.output_activation == ActivationType::SOFTMAX) != (model.loss_type == LossType::CCE)) {
        throw std::invalid_argument("Softmax activation and CCE loss must be paired together");
    }
    if (model.loss_type == LossType::CCE && num_classes < 2) {
        throw std::invalid_argument("Softmax activation requires at least 2 output classes");
    }
    if (model.loss_type == LossType::BCE && model.output_activation != ActivationType::SIGMOID) {
        throw std::invalid_argument("BCE loss is only compatible with Sigmoid activation");
    }
}

// Add the passed layer to the network's layer list
void Network::addLayer(std::unique_ptr<Layer> layer) {
    this->layers.emplace_back(std::move(layer));
}

// Utility function to get all DenseLayer pointers from the network
std::vector<const DenseLayer*> Network::getDenseLayers() const {
    std::vector<const DenseLayer*> dense_layers;
    for (const auto& layer : layers) {
        if (auto* dense_layer = dynamic_cast<const DenseLayer*>(layer.get())) {
            dense_layers.push_back(dense_layer);
        }
    }
    return dense_layers;
}

// Snapshot the current weights and biases of all layers in the network
void Network::snapshotParameters() {
    for (auto& layer : layers) {
        layer->takeSnapshot();
    }
}

// Restore the weights and biases of all layers in the network to the last snapshot
void Network::restoreParameters() {
    for (auto& layer : layers) {
        layer->restoreSnapshot();
    }
}

// Set the normalizers for input and target data
void Network::setNormalizers(Normalizer input, Normalizer target) {
    features_norm = std::move(input);
    labels_norm = std::move(target);
}

// Inference forward pass, does not record the input of each layer, since backward will not be called
Matrix Network::predict(const Matrix& input) const {
    Matrix out = features_norm.apply(input);
    for (const auto& layer : layers) {
        out = layer->forward(out, NetworkMode::TEST);
    }
    // Revert the normalization of the output to return predictions in the original scale
    return labels_norm.revert(out);
}

// Training forward pass, records the input of each layer for use in the backward pass
Matrix Network::forward(const Matrix& input) {
    Matrix out = features_norm.apply(input);
    for (auto& layer : layers) {
        out = layer->forward(out, NetworkMode::TRAIN);
    }
    // Revert the normalization of the output to return predictions in the original scale
    return labels_norm.revert(out);
}

// Training backward pass, propagates the gradient from the error to the first layer.
void Network::backward(const Matrix& output_gradient, Scalar decay_fraction) {
    Matrix gradient = output_gradient;
    for (auto& layer : layers | std::views::reverse) {
        const bool is_first_layer = (&layer == &layers.front());
        gradient = layer->backward(gradient, model, decay_fraction, is_first_layer);
    }
}

// Train the network on the training split and evaluate on the holdout split
RunCurves Network::train(const Dataset& dataset, const DataSplit& indices, const TrainContext& ctx) {
    // Initial time for the run
    const auto run_start = std::chrono::steady_clock::now();

    // Run constants
    const bool is_first_run = (!ctx.in_model_selection || ctx.inner_index == 0) && ctx.trial == 0;

    // Logging and metrics variables
    RunCurves curves(model.task);
    int logged_epoch = 0;

    // Validate the split
    const int input_size = static_cast<int>(indices.train_indices.size());
    if (input_size == 0) {
        throw std::runtime_error("Split has an empty training set");
    }

    // Extract features and labels from the dataset
    const auto train_features = dataset.features(Eigen::placeholders::all, indices.train_indices);
    const auto train_labels = dataset.labels(Eigen::placeholders::all, indices.train_indices);
    const auto holdout_features = dataset.features(Eigen::placeholders::all, indices.test_indices);
    const auto holdout_labels = dataset.labels(Eigen::placeholders::all, indices.test_indices);

    /*
    The normalizers are fitted on the training data and then applied to both the training and holdout data
    This avoids data leakage and ensures that the data is transformed in the same way
    For classification tasks, the labels are not normalized
    */
    features_norm = Normalizer::fit(model.norm_type, train_features);
    labels_norm = model.task == TaskType::REGRESSION ? Normalizer::fit(model.norm_type, train_labels) : Normalizer{};

    // Calculate the number of batches
    const int current_batch_size = (model.batch_size == 0) ? input_size : model.batch_size;
    const int num_batches = (input_size + current_batch_size - 1) / current_batch_size;

    // Logging setup
    std::ofstream log_file;
    auto flush_log = [&](int upto_epoch) {
        const std::string log_filename = std::format("{}/outer{}_{}m{}.csv", MODEL_PATH, ctx.outer_index, ctx.in_model_selection ? "inner_" : "", ctx.model_id);

        if (!log_file.is_open()) {
            log_file.open(log_filename, is_first_run ? std::ios::trunc : std::ios::app);
            if (!log_file.is_open()) {
                throw std::runtime_error("Failed to open log file: " + log_filename);
            }
            if (is_first_run) {
                log_file << "fold,trial,epoch,train_error,test_error,train_mse,test_mse,train_mee,test_mee,train_acc,test_acc\n";
            }
        }
        // Write the metrics for each epoch up to the specified epoch
        const int available = static_cast<int>(std::min(curves.training.size(), curves.holdout.size()));
        const int last = std::min(upto_epoch, available - 1);
        while (logged_epoch <= last) {
            const Metrics& train = curves.training.epochs[logged_epoch];
            const Metrics& holdout = curves.holdout.epochs[logged_epoch];
            log_file << ctx.inner_index << ","
                     << ctx.trial << ","
                     << logged_epoch << ","
                     << train.error << ","
                     << holdout.error << ","
                     << train.mse << ","
                     << holdout.mse << ","
                     << train.mee << ","
                     << holdout.mee << ","
                     << train.accuracy * 100.0 << ","
                     << holdout.accuracy * 100.0 << "\n";
            logged_epoch++;
        }
        log_file.flush();
    };

    // LEARNING RATE
    /*
        The number of epochs is derived from the total weight updates and the batch size, since smaller batch sizes result in more updates per epoch
        This ensures warmup, decay, and early stopping are applied consistently and meaningfully regardless of the batch size used

        initial_eta is the predefined starting learning rate
        target_eta is the fraction of it (TARGET_ETA_MULTIPLIER) that is reached after the decay period, `tau` epochs (TAU_MULTIPLIER)

        WARMUP:
        eta = initial_eta * (epoch / warmup_epochs)
        Learning rate linearly increases from `initial_eta / warmup_epochs` to `initial_eta` over `warmup_epochs` number of epochs

        LINEAR DECAY:
        eta = initial_eta * (1 - (epoch - warmup_epochs) / tau) + target_eta * ((epoch - warmup_epochs) / tau)
        Learning rate linearly decreases from `initial_eta` to `target_eta` over `tau` number of epochs, immediately following the warmup period
    */
    const int epochs = epochs_for(ctx.updates, model.batch_size, input_size);
    const int warmup_epochs = std::clamp(ctx.warmup / num_batches, 0, std::max(0, epochs - 1));
    const Scalar initial_eta = model.eta;
    const Scalar target_eta = initial_eta * TARGET_ETA_MULTIPLIER;
    const Scalar tau = std::max(Scalar(1), static_cast<Scalar>(epochs - warmup_epochs) * TAU_MULTIPLIER);
    // ----------------------

    // --- EARLY STOPPING ---
    /*
        NONE:
        Trains for the full number of epochs
        Used when explicitly selected, or as a fallback if other rules cannot be applied

        ERROR:
        Stops training when a predefined `target error` threshold is reached
        If no target is available, it falls back to PATIENCE
        - In model selection, the holdout (validation) error is used as the target
        - Everywhere else, the training error is used since the holdout (assessment) set cannot be used

        PATIENCE:
        Stops training when the error has not improved for a set number of epochs
        - In model selection, improvement is checked on the holdout (validation) set
        - Everywhere else, improvement is checked on the training set since the holdout (assessment) set cannot be used
    */
    Scalar patience_error = 0.0;
    int epochs_without_improvement = 0;
    bool auto_early_stop_flag = false;
    const StoppingRule rule = [&ctx] {
        if (ctx.stopping == StoppingRule::ERROR && ctx.target_error) {
            return StoppingRule::ERROR;
        }
        return (ctx.stopping != StoppingRule::NONE && ctx.patience > 0) ? StoppingRule::PATIENCE : StoppingRule::NONE;
    }();
    // To make the stopping rules consistent across different batch sizes, the patience is converted from weight updates to epochs, since smaller batch sizes result in more weight updates per epoch
    const int patience_epochs = std::clamp(ctx.patience / num_batches, 1, std::max(1, epochs));
    // --------------------

    // Reused the same buffer for indices to avoid repeated allocations
    std::vector<int> batch_indices;
    batch_indices.reserve(current_batch_size);
    std::vector<int> epoch_indices = indices.train_indices;

    // Main training loop over epochs
    for (int i = 0; i < epochs; i++) {
        std::ranges::shuffle(epoch_indices, get_trial_generator());

        // --- Learning rate logic ---
        if (i < warmup_epochs) {
            // Warmup: linearly increase the learning rate from initial_eta/warmup_epochs to initial_eta
            model.eta = initial_eta * (static_cast<Scalar>(i + 1) / static_cast<Scalar>(warmup_epochs));
        } else {
            // Decay: linearly decrease the learning rate from initial_eta to target_eta
            const Scalar gamma = std::min(Scalar(1), static_cast<Scalar>(i - warmup_epochs) / tau);
            model.eta = (Scalar(1) - gamma) * initial_eta + (gamma * target_eta);
        }
        // ---------------------------

        // --- Training loop over batches ---
        LearningCurve batch_curve;
        for (int j = 0; j < num_batches && !early_stop_flag && !finish_flag; j++) {
            // Determine the starting index and size for the current batch
            const int index_start = j * current_batch_size;
            const int actual_batch_size = std::min(index_start + current_batch_size, input_size) - index_start;

            // Extract the current batch of features and labels
            batch_indices.assign(epoch_indices.begin() + index_start, epoch_indices.begin() + index_start + actual_batch_size);
            Matrix batch_features = dataset.features(Eigen::placeholders::all, batch_indices);
            Matrix batch_labels = dataset.labels(Eigen::placeholders::all, batch_indices);

            Matrix batch_prediction = forward(batch_features);
            batch_curve.add_batch(batch_labels, batch_prediction, model.loss_type, curves.track_accuracy());

            const Scalar decay_fraction = static_cast<Scalar>(actual_batch_size) / static_cast<Scalar>(input_size);
            backward(loss_pair.derivative(labels_norm.apply(batch_labels), labels_norm.apply(batch_prediction)), decay_fraction);
        }
        curves.training.append_epochs(batch_curve);
        curves.training.normalize_epoch();

        // Check for NaN or Inf in the training error
        if (!std::isfinite(curves.training.last_error())) {
            curves.training.invalid = true;
            curves.holdout.invalid = true;
            std::println("\n[Forced Early stopping (NaN/Inf): Inner Fold {} | Outer Fold {} | Epoch {}]", ctx.in_model_selection ? ctx.inner_index : -1, ctx.outer_index, i);
            break;
        }
        // Evaluate the holdout set if needed
        const bool needs_holdout = (rule == StoppingRule::PATIENCE && ctx.in_model_selection) || ctx.logging;
        const bool predict_holdout = needs_holdout && !indices.test_indices.empty();
        if (predict_holdout) {
            Matrix holdout_prediction = predict(holdout_features);
            curves.holdout.append_epoch(holdout_labels, holdout_prediction, model.loss_type, curves.track_accuracy());
        }
        // ----------------------------------

        //  --- Early stopping logic ---
        if (rule == StoppingRule::ERROR) {
            // Stop training if the training error has reached the target error level
            if (curves.training.last_error() <= *ctx.target_error) {
                auto_early_stop_flag = true;
            }
        } else if (rule == StoppingRule::PATIENCE) {
            // Pick the correct error to check for improvement based on whether we are in model selection or not
            Scalar current_error = ctx.in_model_selection ? curves.holdout.last_error() : curves.training.last_error();
            // If the current error is better than the best error so far (with tolerance), reset the patience counter and save the model parameters
            if (i == 0 || current_error < patience_error - std::abs(patience_error) * ES_TOLERANCE) {
                epochs_without_improvement = 0;
                patience_error = current_error;
                curves.best_epoch = i;
                snapshotParameters(); // Save the current parameters as the best epoch
            } else {
                epochs_without_improvement++;
            }

            if (epochs_without_improvement >= patience_epochs) {
                auto_early_stop_flag = true;
            }
        }
        // ----------------------------

        // Log the metrics for the current epoch to the log file
        if (ctx.logging && (i % LOG_FREQ == 0 || i == epochs - 1)) {
            flush_log(i);
        }

        if (early_stop_flag || finish_flag || auto_early_stop_flag) {
            if (early_stop_flag || finish_flag) {
                std::println(stderr, "\n[Manual Early stopping: Inner Fold {} | Outer Fold {} | Epoch {}]", ctx.in_model_selection ? ctx.inner_index : -1, ctx.outer_index, i);
            }
            break;
        }
    }

    // Roll back to the parameters of the best epoch
    if (rule == StoppingRule::PATIENCE) {
        restoreParameters();
    } else {
        // The other rules never snapshot, so the run simply ends on the last epoch it reached
        curves.best_epoch = std::max(0, static_cast<int>(curves.training.size()) - 1);
    }

    // Write final epochs to the log file
    if (ctx.logging) {
        flush_log(epochs - 1);
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    model.eta = initial_eta; // Restore the configured learning rate after decay
    early_stop_flag = 0;     // Reset the early stop flag for the next fold

    // Total time for the run
    curves.duration = std::chrono::duration<Scalar>(std::chrono::steady_clock::now() - run_start).count();

    return curves;
}