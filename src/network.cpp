#include "network.hpp"

#include "dataset.hpp"
#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <print>
#include <string>
#include <vector>

// Network constructor that initializes model parameters and sets the loss function
Network::Network(const Model& model) : model(model) {
    setLossFunction(model.loss_type);
}

// Network constructor that calls the layers building function with the specified number of features and classes
Network::Network(const Model& model, int num_features, int num_classes) : Network(model) {
    buildLayers(model, num_features, num_classes, nullptr, nullptr, true);
}

// Network constructor that calls the layers building function with the specified weights and biases for each layer
Network::Network(const Model& model, const std::vector<Matrix>& weights, const std::vector<Vector>& biases, bool instantiate_optimizer) : Network(model) {
    if (weights.size() != biases.size() || weights.size() != model.net_struct.size() + 1) {
        throw std::invalid_argument("Mismatch between the number of weights, biases, and layers in the model");
    }
    buildLayers(model, static_cast<int>(weights.front().cols()), static_cast<int>(weights.back().rows()), &weights, &biases, instantiate_optimizer);
}

// Function that builds each layer of the network, both loaded and initialized
void Network::buildLayers(const Model& model, int num_features, int num_classes, const std::vector<Matrix>* weights, const std::vector<Vector>* biases, bool instantiate_optimizer) {
    validateModel(model, num_features, num_classes);

    for (size_t i = 0; i < model.net_struct.size(); ++i) {
        // Add a DenseLayer and an ActivationLayer for each layer in the network structure
        if (weights != nullptr) {
            addLayer(std::make_unique<DenseLayer>((*weights)[i], (*biases)[i], model.opt_type, instantiate_optimizer));
        } else {
            const int input_features = (i == 0) ? num_features : model.net_struct[i - 1];
            addLayer(std::make_unique<DenseLayer>(input_features, model.net_struct[i], model.init_type, model.opt_type));
        }
        addLayer(std::make_unique<ActivationLayer>(model.hidden_activation));
    }

    // Add the final output layer with the specified number of classes and output activation
    if (weights != nullptr) {
        addLayer(std::make_unique<DenseLayer>(weights->back(), biases->back(), model.opt_type, instantiate_optimizer));
    } else {
        addLayer(std::make_unique<DenseLayer>(model.net_struct.back(), num_classes, model.init_type, model.opt_type));
    }

    // Add the output activation layer. Only this layer can have its derivative handled by the loss
    const bool derivative_in_loss = LossFunctions::includes_output_derivative(model.loss_type);
    addLayer(std::make_unique<ActivationLayer>(model.output_activation, derivative_in_loss));
}

// Utility function to set the loss function and its derivative based on the specified LossType
void Network::setLossFunction(LossType lossType) {
    try {
        loss_func = Maps::loss_map.at(lossType).first;
        loss_derivative = Maps::loss_map.at(lossType).second;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Unsupported loss function type");
    }
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
    if (model.task == TaskType::REGRESSION && model.loss_type != LossType::MSE) {
        throw std::invalid_argument("Regression tasks must use MSE loss");
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
        layer->snapshot();
    }
}

// Restore the weights and biases of all layers in the network to the last snapshot
void Network::restoreParameters() {
    for (auto& layer : layers) {
        layer->restore();
    }
}

// Inference forward pass, does not record the input of each layer, since backward will not be called (can be made const)
Matrix Network::predict(const Matrix& input) const {
    Matrix out = input;
    for (const auto& layer : layers) {
        out = layer->forward(out, false);
    }
    return out;
}

// Training forward pass, records the input of each layer for use in the backward pass
Matrix Network::forward(const Matrix& input) {
    Matrix out = input;
    for (auto& layer : layers) {
        out = layer->forward(out, true);
    }
    return out;
}

// Training backward pass, propagates the gradient from the loss to the first layer
void Network::backward(const Matrix& output_gradient) {
    Matrix gradient = output_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        const bool is_first_layer = (std::next(it) == layers.rend());
        gradient = (*it)->backward(gradient, model, is_first_layer);
    }
}

// Train the network using the dataset provided by the model_set
SplitResults Network::train(const Dataset& dataset, const DataSplit& indices, const TrainContext& ctx) {
    // File logging handling
    std::ofstream log_file;
    if (ctx.logging) {
        std::string log_filename = std::format("{}/outer{}_{}m{}.csv", MODEL_PATH, ctx.outer_index, ctx.in_model_selection ? "inner_" : "", ctx.model_id);
        bool is_first_write = !ctx.in_model_selection || ctx.inner_index == 0;
        log_file.open(log_filename, is_first_write ? std::ios::trunc : std::ios::app);
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file for writing: " + log_filename);
        }
        if (is_first_write) {
            log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";
        }
    }

    // Validate the split
    const int input_size = static_cast<int>(indices.train_indices.size());
    if (input_size == 0 || indices.test_indices.empty()) {
        throw std::runtime_error("Split has an empty training or test set");
    }

    // Extract test features and labels from the dataset
    Matrix test_features = dataset.features(Eigen::placeholders::all, indices.test_indices);
    Matrix test_labels = dataset.labels(Eigen::placeholders::all, indices.test_indices);

    // Calculate the number of batches
    const int current_batch_size = (model.batch_size == 0) ? input_size : model.batch_size;
    const int num_batches = (input_size + current_batch_size - 1) / current_batch_size;

    // Early Stopping variables
    Scalar patience_loss = 0.0;
    int epochs_without_improvement = 0;
    bool auto_early_stop_flag = false;

    // Logging and metrics variables
    SplitResults split_results;
    split_results.task = model.task;
    int logged_epoch = 0;
    const bool track_accuracy = (model.task == TaskType::CLASSIFICATION);

    // Flush every epoch up to log file
    auto flush_log = [&](int upto_epoch) {
        if (!log_file.is_open()) {
            return;
        }
        // Sync the logged epoch with the available metrics
        const int available = static_cast<int>(std::min(split_results.train_metrics.size(), split_results.test_metrics.size()));
        const int last = std::min(upto_epoch, available - 1);
        while (logged_epoch <= last) {
            const EpochMetric& train = split_results.train_metrics.epochs[logged_epoch];
            const EpochMetric& test = split_results.test_metrics.epochs[logged_epoch];
            log_file << logged_epoch << ","
                     << train.loss << ","
                     << test.loss << ","
                     << train.accuracy * 100.0 << ","
                     << test.accuracy * 100.0 << "\n";
            logged_epoch++;
        }
        log_file.flush();
    };

    // Linear learning rate decay parameters
    const Scalar initial_eta = model.eta;
    const Scalar target_eta = initial_eta * TARGET_ETA_MULTIPLIER;
    const Scalar tau = std::max(Scalar(1), static_cast<Scalar>(ctx.epochs) * TAU_MULTIPLIER);

    // Reusable batch index buffer and local copy of the training indices to shuffle each epoch
    std::vector<int> batch_indices;
    batch_indices.reserve(current_batch_size);
    std::vector<int> epoch_indices = indices.train_indices;

    for (int i = 0; i < ctx.epochs; i++) {
        // Shuffle the training data indices for current epoch
        std::ranges::shuffle(epoch_indices, get_random_generator());

        // Update the learning rate based on the current epoch using linear decay
        const Scalar gamma = std::min(Scalar(1), static_cast<Scalar>(i) / tau);
        model.eta = (Scalar(1) - gamma) * initial_eta + (gamma * target_eta);

        MetricsResult batch_metrics;
        for (int j = 0; j < num_batches && !early_stop_flag; j++) {
            // Determine the starting index and size for the current batch
            const int index_start = j * current_batch_size;
            const int actual_batch_size = std::min(index_start + current_batch_size, input_size) - index_start;

            // Extract the current batch of features and labels
            batch_indices.assign(epoch_indices.begin() + index_start, epoch_indices.begin() + index_start + actual_batch_size);
            Matrix batch_features = dataset.features(Eigen::placeholders::all, batch_indices);
            Matrix batch_labels = dataset.labels(Eigen::placeholders::all, batch_indices);

            // Forward pass to get predictions for the current batch
            Matrix batch_prediction = forward(batch_features);

            // Calculate the loss and accuracy for the current batch and accumulate for the epoch
            batch_metrics.add(batch_labels, batch_prediction, loss_func, track_accuracy);

            // Start the backward pass to update weights and biases based on the output loss gradient
            backward(loss_derivative(batch_labels, batch_prediction));
        }
        split_results.train_metrics.append(batch_metrics);
        split_results.train_metrics.average_last();

        // Check for NaN or Inf in the latest training loss to stop training
        if (!std::isfinite(split_results.train_metrics.last_loss())) {
            split_results.train_metrics.invalid = true;
            split_results.test_metrics.invalid = true;
            std::println("\n[Forced Early stopping (NaN/Inf): Inner Fold {} | Outer Fold {} | Epoch {}]", ctx.in_model_selection ? ctx.inner_index : -1, ctx.outer_index, i);
            break;
        }

        // Evaluate the model on the test set and calculate the loss and accuracy
        const bool need_test_metrics = (ctx.patience > 0 && ctx.in_model_selection) || ctx.logging;
        if (need_test_metrics) {
            Matrix test_prediction = predict(test_features);
            split_results.test_metrics.append(test_labels, test_prediction, loss_func, track_accuracy);
        }

        // Early stopping logic based on the patience parameter
        if (ctx.patience > 0) {
            Scalar current_loss = ctx.in_model_selection ? split_results.test_metrics.last_loss() : split_results.train_metrics.last_loss();
            // If the current loss is better than the best loss so far (with tolerance), reset the patience counter and save the model parameters
            if (i == 0 || current_loss < patience_loss - std::abs(patience_loss) * ES_REL_TOL) {
                epochs_without_improvement = 0;
                patience_loss = current_loss;
                snapshotParameters(); // Save the current parameters as the best epoch
            } else {
                epochs_without_improvement++;
            }

            if (epochs_without_improvement >= ctx.patience) {
                auto_early_stop_flag = true;
            }
        }

        // Log the metrics for the current epoch to the log file
        if (ctx.logging && (i % LOG_FREQ == 0 || i == ctx.epochs - 1)) {
            flush_log(i);
        }

        if (early_stop_flag || auto_early_stop_flag) {
            if (early_stop_flag) {
                std::println("\n[Manual Early stopping: Inner Fold {} | Outer Fold {} | Epoch {}]", ctx.in_model_selection ? ctx.inner_index : -1, ctx.outer_index, i);
            } else {
                std::println("\n[Automatic Early stopping: Inner Fold {} | Outer Fold {} | Epoch {}]", ctx.in_model_selection ? ctx.inner_index : -1, ctx.outer_index, i);
            }
            break;
        }
    }

    // Roll back to the parameters of the best epoch if early stopping was triggered automatically
    if (auto_early_stop_flag) {
        restoreParameters();
    }

    // Write final epochs to the log file
    flush_log(ctx.epochs - 1);

    model.eta = initial_eta; // Restore the configured learning rate after decay
    early_stop_flag = 0;     // Reset the early stop flag for the next fold
    if (log_file.is_open()) {
        log_file.close();
    }
    return split_results;
}
