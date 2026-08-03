#include "network.hpp"

#include "dataset.hpp"
#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <fstream>
#include <functional>
#include <memory>
#include <print>
#include <random>
#include <string>
#include <vector>

// -- DenseLayer class implementation --

// DenseLayer constructor that initializes weights based on the specified initialization type
DenseLayer::DenseLayer(int input_size, int output_size, InitType init_type, OptimizerType opt_type) {
    // Determine the distribution value based on the initialization type
    Scalar distribution_value;
    switch (init_type) {
    case InitType::RANDOM:
        distribution_value = 1.0;
        break;
    case InitType::LECUN:
        distribution_value = std::sqrt(1.0 / static_cast<Scalar>(input_size));
        break;
    case InitType::GLOROT:
        distribution_value = std::sqrt(6.0 / static_cast<Scalar>(input_size + output_size));
        break;
    case InitType::HE:
        distribution_value = std::sqrt(2.0 / static_cast<Scalar>(input_size));
        break;
    default:
        throw std::invalid_argument("Unsupported initialization type");
    }

    // Initialize weights and biases pre-transposed for more efficient matrix multiplication later on
    switch (init_type) {
    case InitType::HE: {
        std::normal_distribution<Scalar> normal_dist(0.0, distribution_value);
        W = Matrix::NullaryExpr(output_size, input_size, [&]() { return normal_dist(get_random_generator()); });
    } break;
    default: {
        std::uniform_real_distribution<Scalar> uniform_dist(-distribution_value, distribution_value);
        W = Matrix::NullaryExpr(output_size, input_size, [&]() { return uniform_dist(get_random_generator()); });
    } break;
    }

    setOptimizer(opt_type);
    b = Vector::Zero(output_size);
}

// Forward pass through the DenseLayer, saving input and output for backpropagation
Matrix DenseLayer::forward(const Matrix& input_matrix, bool training) {
    if (W.cols() != input_matrix.rows()) {
        throw std::runtime_error("Dimension mismatch: " + std::to_string(W.cols()) + " features expected, but " + std::to_string(input_matrix.rows()) + " features provided");
    }
    Matrix output_matrix = (W * input_matrix).colwise() + b; // Multiply weights with input and add bias
    if (training) {
        X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
    }
    return output_matrix;
}

// Backward pass through the DenseLayer, updating weights and biases based on the output gradient
Matrix DenseLayer::backward(const Matrix& output_gradient, const Model& model) {
    Matrix weights_delta = (output_gradient * X.transpose()) / X.cols(); // Calculate the delta of weights (average over the batch)
    Vector bias_delta = output_gradient.rowwise().sum() / X.cols();      // Calculate the delta of biases (sum over columns to aggregate, and average over the batch)
    Matrix input_gradient = W.transpose() * output_gradient;             // Calculate the neuron gradient to propagate to the previous layer

    // Update weights and biases using the optimizer
    optimizer->update(W, b, weights_delta, bias_delta, model);

    return input_gradient;
}

// -- ActivationLayer class implementation --

// ActivationLayer constructor that initializes the activation function and its derivative based on the specified ActivationType
ActivationLayer::ActivationLayer(ActivationType activationType) {
    try {
        activation = Maps::activation_map.at(activationType).first;
        activation_derivative = Maps::activation_map.at(activationType).second;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Unsupported activation function type");
    }
}

// Forward pass through the ActivationLayer, applying the activation function to the input matrix
Matrix ActivationLayer::forward(const Matrix& input_matrix, bool training) {
    Matrix output_matrix = activation(input_matrix); // Call the activation function on the input matrix
    if (training) {
        X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
    }
    return output_matrix;
}

// Backward pass through the ActivationLayer, calculating the gradient with respect to the input
Matrix ActivationLayer::backward(const Matrix& output_gradient, [[maybe_unused]] const Model& model) {
    Matrix derivative = activation_derivative(X);    // Calculate the derivative of the activation function with respect to the input
    return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
}

// -- Network class implementation --

// Network constructor that initializes args and sets the loss function
Network::Network(const Model& model) {
    this->model = model;
    setLossFunction(model.loss_type);
}

// Network constructor that initializes pairs of DenseLayer and ActivationLayer based on the defined network structure
Network::Network(const Model& model, const int num_features, const int num_classes) : Network(model) {
    for (size_t i = 0; i < model.net_struct.size(); ++i) {
        int input_features = i == 0 ? num_features : model.net_struct[i - 1];
        int num_neurons = model.net_struct[i];

        // Add a DenseLayer and an ActivationLayer for each layer in the network structure
        addLayer(std::make_unique<DenseLayer>(input_features, num_neurons, model.init_type, model.opt_type));
        addLayer(std::make_unique<ActivationLayer>(model.hidden_activation));
    }
    // Add the final output layer with the specified number of classes and output activation
    addLayer(std::make_unique<DenseLayer>(model.net_struct.back(), num_classes, model.init_type, model.opt_type));
    addLayer(std::make_unique<ActivationLayer>(model.output_activation));

    validateNetworkStructure(num_classes);
}

// Network constructor that initializes pairs of DenseLayer and ActivationLayer with loaded weights and biases
Network::Network(const Model& model, std::vector<Matrix> weights, std::vector<Vector> biases) : Network(model) {
    for (size_t i = 0; i < model.net_struct.size(); ++i) {
        // Add a DenseLayer and an ActivationLayer for each layer in the network structure with loaded weights and biases
        addLayer(std::make_unique<DenseLayer>(weights[i], biases[i], model.opt_type));
        addLayer(std::make_unique<ActivationLayer>(model.hidden_activation));
    }
    // Add the final output layer with the specified number of classes and output activation with loaded weights and biases
    addLayer(std::make_unique<DenseLayer>(weights.back(), biases.back(), model.opt_type));
    addLayer(std::make_unique<ActivationLayer>(model.output_activation));

    validateNetworkStructure(weights.back().rows());
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

void Network::validateNetworkStructure(int num_classes) const {
    if (layers.empty()) {
        throw std::invalid_argument("Network is empty");
    }
    if (model.task == TaskType::REGRESSION && model.loss_type != LossType::MSE) {
        throw std::invalid_argument("Regression tasks must use MSE loss");
    }
    if ((model.output_activation == ActivationType::SOFTMAX) != (model.loss_type == LossType::CCE)) {
        // Must be paired together because Softmax derivative is combined with CCE loss derivative to simplify the calculation
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
        if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
            dense_layers.push_back(dense_layer);
        }
    }
    return dense_layers;
}

// Forward pass through the network, saving weights norm for loss calculation with regularization
Matrix Network::predict(const Matrix& out, bool training) {
    weights_norm = 0.0;
    Matrix new_out = out;
    for (auto& layer : layers) {
        new_out = layer->forward(new_out, training); // Forward pass through each layer
        weights_norm += layer->getWeightNorm();      // Accumulate the squared norm of weights for L2 regularization loss (not really used anymore)
    }
    return new_out;
}

// Train the network using the dataset provided by the model_set
SplitResults Network::train(const Dataset& dataset, const DataSplit& indices, int epochs, int patience, int model_index, int outer_index, int inner_index, bool logging) {
    if (outer_index == 0 && inner_index == 0) {
        model.print();
    }
    bool in_model_selection = inner_index != -1; // if there are no inner folds, then we are not in model selection mode

    // File logging handling
    std::ofstream log_file;
    if (logging) {
        std::string log_filename = std::format("{}/outer{}_{}m{}.csv", MODEL_PATH, outer_index, inner_index >= 0 ? "inner_" : "", model_index);
        bool is_first_write = (inner_index <= 0);
        log_file.open(log_filename, is_first_write ? std::ios::trunc : std::ios::app);
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file for writing: " + log_filename);
        }
        if (is_first_write) {
            log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";
        }
    }

    // Extract test features and labels from the dataset
    Matrix test_features = dataset.features(Eigen::placeholders::all, indices.test_indices);
    Matrix test_labels = dataset.labels(Eigen::placeholders::all, indices.test_indices);

    // Calculate the number of batches
    const int input_size = indices.train_indices.size();
    if (input_size == 0) {
        throw std::runtime_error("Training set is empty.");
    }
    const int current_batch_size = (model.batch_size == 0) ? input_size : model.batch_size;
    const int num_batches = (input_size + current_batch_size - 1) / current_batch_size;

    // Early Stopping variables
    Scalar patience_loss = 0.0;
    int epochs_without_improvement = 0;
    bool auto_early_stop_flag = false;

    // Logging and metrics variables
    SplitResults split_results;
    int logged_epoch = 0;

    // Linear learning rate decay parameters
    Scalar initial_eta = model.eta;
    Scalar target_eta = initial_eta * TARGET_ETA_MULTIPLIER;
    Scalar tau = epochs * TAU_MULTIPLIER;

    // Local copy of the training indices to shuffle for each epoch
    std::vector<int> epoch_indices = indices.train_indices;
    for (int i = 0; i < epochs; i++) {
        // Shuffle the training data indices for current epoch
        std::ranges::shuffle(epoch_indices, get_random_generator());

        // Update the learning rate based on the current epoch using linear decay
        Scalar gamma = std::min(static_cast<Scalar>(1.0), static_cast<Scalar>(i / tau));
        model.eta = (1.0 - gamma) * initial_eta + (gamma * target_eta);

        MetricsResult batch_metrics;
        for (int j = 0; j < num_batches && !early_stop_flag; j++) {
            // Determine the start and end indices for the current batch
            const int index_start = j * current_batch_size;
            const int index_end = std::min(index_start + current_batch_size, input_size);

            // Extract the current batch of features and labels
            std::vector<int> batch_indices(epoch_indices.begin() + index_start, epoch_indices.begin() + index_end);
            Matrix batch_features = dataset.features(Eigen::placeholders::all, batch_indices);
            Matrix batch_labels = dataset.labels(Eigen::placeholders::all, batch_indices);

            // Forward pass to get predictions for the current batch
            Matrix batch_prediction = predict(batch_features, true);

            // Calculate the loss and accuracy for the current batch and accumulate for the epoch
            batch_metrics.add(batch_labels, batch_prediction, loss_func);

            // Start the backward pass to update weights and biases based on the output loss gradient
            Matrix gradient = loss_derivative(batch_labels, batch_prediction);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradient = (*it)->backward(gradient, model);
            }
        }
        split_results.train_metrics.append(batch_metrics);

        if ((patience > 0 && in_model_selection) || logging) {
            // Evaluate the model on the test set and calculate the loss and accuracy
            Matrix test_prediction = predict(test_features);
            split_results.test_metrics.append(test_labels, test_prediction, loss_func);
        }

        split_results.average(); // Average batch metrics for the epoch (if batch training was used)

        // Early stopping logic based on the patience parameter
        if (patience > 0) {
            if (in_model_selection) {
                // Use validation loss (that stopped decreasing) for early stopping during model selection (using patience)
                Scalar test_loss = split_results.test_metrics.loss.back();
                if (i == 0 || test_loss < patience_loss) {
                    epochs_without_improvement = 0;
                    patience_loss = test_loss;
                } else {
                    epochs_without_improvement++;
                }

                if (epochs_without_improvement >= patience) {
                    auto_early_stop_flag = true;
                }

            } else {
                // Use training loss (that stopped decreasing significantly) for early stopping during final training (using patience)
                Scalar train_loss = split_results.train_metrics.loss.back();
                int train_patience = std::max(5, patience / 2); // Use a smaller patience for this method
                if (i > train_patience) {
                    patience_loss = split_results.train_metrics.loss[i - train_patience]; // Compare with loss from 'train_patience' epochs ago
                    Scalar progress = (patience_loss - train_loss) / (patience_loss + EPSILON);
                    if (progress < ES_TRAIN) {
                        auto_early_stop_flag = true;
                    }
                }
            }
        }
        if (logging && log_file.is_open()) {
            // Log the metrics for the current epoch to the log file
            if (i > 0 && (i % LOG_FREQ == 0 || i == epochs - 1)) {
                while (logged_epoch < i) {
                    log_file << logged_epoch << ","
                             << split_results.train_metrics.loss[logged_epoch] << ","
                             << split_results.test_metrics.loss[logged_epoch] << ","
                             << split_results.train_metrics.accuracy[logged_epoch] * 100.0 << ","
                             << split_results.test_metrics.accuracy[logged_epoch] * 100.0 << "\n";
                    logged_epoch++;
                }
                log_file.flush();
            }
        }

        if (early_stop_flag || auto_early_stop_flag) {
            if (early_stop_flag) {
                std::println("\n[Manual Early stopping: Inner Fold {} | Outer Fold {} | Epoch {}]", inner_index, outer_index, i);
                early_stop_flag = 0; // Reset the early stop flag for the next fold
            } else {
                std::println("\n[Automatic Early stopping: Inner Fold {} | Outer Fold {} | Epoch {}]", inner_index, outer_index, i);
            }
            break;
        }
    }

    if (log_file.is_open()) {
        log_file.close();
    }
    return split_results;
}