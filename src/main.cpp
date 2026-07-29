#include "cli.hpp"
#include "dataset.hpp"
#include "dump.hpp"
#include "functions.hpp"
#include "network.hpp"
#include "types.hpp"

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

bool TRAIN_MODEL = true;

// -- DenseLayer class implementation --

// DenseLayer constructor that initializes weights based on the specified initialization type
DenseLayer::DenseLayer(int input_size, int output_size, InitializationType init_type) {
    // Determine the distribution value based on the initialization type
    Scalar distribution_value;
    switch (init_type) {
    case InitializationType::RANDOM:
        distribution_value = 1.0;
        break;
    case InitializationType::LECUN:
        distribution_value = std::sqrt(1.0 / static_cast<Scalar>(input_size));
        break;
    case InitializationType::GLOROT:
        distribution_value = std::sqrt(6.0 / static_cast<Scalar>(input_size + output_size));
        break;
    case InitializationType::HE:
        distribution_value = std::sqrt(2.0 / static_cast<Scalar>(input_size));
        break;
    default:
        throw std::invalid_argument("Unsupported initialization type.");
    }

    // Initialize weights and biases pre-transposed for efficient matrix multiplication
    switch (init_type) {
    case InitializationType::HE: {
        std::normal_distribution<Scalar> normal_dist(0.0, distribution_value);
        W = Matrix::NullaryExpr(output_size, input_size, [&]() { return normal_dist(get_random_generator()); });
    } break;
    default: {
        std::uniform_real_distribution<Scalar> uniform_dist(-distribution_value, distribution_value);
        W = Matrix::NullaryExpr(output_size, input_size, [&]() { return uniform_dist(get_random_generator()); });
    } break;
    }

    b = Vector::Zero(output_size);
    delta_W = Matrix::Zero(output_size, input_size);
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
Matrix DenseLayer::backward(const Matrix& output_gradient, const Args& args) {
    Matrix weights_delta = (output_gradient * X.transpose()) / args.batch_size; // Calculate the delta of weights (average over the batch)
    Vector bias_delta = output_gradient.rowwise().sum() / args.batch_size;      // Calculate the delta of biases (sum over columns to aggregate, and average over the batch)
    Matrix input_gradient = W.transpose() * output_gradient;                    // Calculate the neuron gradient to propagate to the previous layer

    delta_W = -args.eta * weights_delta + args.alpha * delta_W; // Update delta_W with learning rate and momentum
    Matrix l2_penalty = args.eta * args.lambda * W;             // L2 regularization term

    W = W + delta_W - l2_penalty; // Update weights with L2 regularization and momentum
    b -= args.eta * bias_delta;   // Update biases with learning rate

    return input_gradient;
}

// -- ActivationLayer class implementation --

// ActivationLayer constructor that initializes the activation function and its derivative based on the specified ActivationType
ActivationLayer::ActivationLayer(ActivationType activationType) {
    try {
        activation = activation_map.at(activationType).first;
        activation_derivative = activation_map.at(activationType).second;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Unsupported activation function type.");
    }
}

// Forward pass through the ActivationLayer, applying the activation function to the input matrix
Matrix ActivationLayer::forward(const Matrix& input_matrix, bool training) {
    Matrix output_matrix = activation(input_matrix); // Apply the activation function to the input matrix
    if (training) {
        X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
    }
    return output_matrix;
}

// Backward pass through the ActivationLayer, calculating the gradient with respect to the input
Matrix ActivationLayer::backward(const Matrix& output_gradient, [[maybe_unused]] const Args& args) {
    Matrix derivative = activation_derivative(X);    // Calculate the derivative of the activation function with respect to the input
    return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
}

// -- Network class implementation --

// Network constructor that initializes args and sets the loss function
Network::Network(const Args& cli_args) {
    args = cli_args;

    // SOFTMAX activation is used with Categorical Cross-Entropy loss, while other activations use Mean Squared Error loss
    if (args.output_activation == ActivationType::SOFTMAX) {
        setLossFunction(LossType::CCE);
    } else {
        setLossFunction(LossType::MSE);
    }
}

// Network constructor that initializes pairs of DenseLayer and ActivationLayer based on the defined network structure
Network::Network(const Args& cli_args, const int num_features, const int num_classes) : Network(cli_args) {
    for (size_t i = 0; i < args.net_struct.size(); ++i) {
        int input_features = i == 0 ? num_features : args.net_struct[i - 1];
        int num_neurons = args.net_struct[i];
        addLayer(new DenseLayer(input_features, num_neurons, args.init_type));
        addLayer(new ActivationLayer(args.hidden_activation));
    }
    addLayer(new DenseLayer(args.net_struct.back(), num_classes, args.init_type));
    addLayer(new ActivationLayer(args.output_activation));
}

// Network constructor that initializes pairs or DenseLayer and ActivationLayer with loaded weights and biases
Network::Network(const Args& cli_args, std::vector<Matrix> weights, std::vector<Vector> biases) : Network(cli_args) {
    for (size_t i = 0; i < args.net_struct.size(); ++i) {
        addLayer(new DenseLayer(weights[i], biases[i]));
        addLayer(new ActivationLayer(args.hidden_activation));
    }
    addLayer(new DenseLayer(weights.back(), biases.back()));
    addLayer(new ActivationLayer(args.output_activation));
}

// Utility function to set the loss function and its derivative based on the specified LossType
void Network::setLossFunction(LossType lossType) {
    try {
        loss_func = loss_map.at(lossType).first;
        loss_derivative = loss_map.at(lossType).second;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Unsupported loss function type.");
    }
}

// Add the passed layer to the network's layer list
void Network::addLayer(Layer* layer) {
    this->layers.push_back(std::unique_ptr<Layer>(layer));
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
Matrix Network::predict(Matrix out, bool training) {
    weights_norm = 0.0;
    for (auto& layer : layers) {
        out = layer->forward(out, training);
        weights_norm += layer->weightNorm();
    }
    return out;
}

// Train the network using the dataset provided by the model_set
void Network::train(const ModelSet& model_set) {
    std::cout << std::endl
              << "\nTraining Configuration:" << "\n"
              << std::left << std::setw(25) << " • Epochs:" << args.epochs << "\n"
              << std::left << std::setw(25) << " • Batch Size:" << args.batch_size << "\n"
              << std::left << std::setw(25) << " • Learning Rate:" << args.eta << "\n"
              << std::left << std::setw(25) << " • Regularization:" << args.lambda << "\n"
              << std::left << std::setw(25) << " • Momentum:" << args.alpha << "\n"
              << std::left << std::setw(25) << " • Hidden Activation:" << activation_type_to_string.at(args.hidden_activation) << "\n"
              << std::left << std::setw(25) << " • Output Activation:" << activation_type_to_string.at(args.output_activation) << "\n"
              << std::left << std::setw(25) << " • Weight Init:" << initialization_type_to_string.at(args.init_type) << "\n";

    // Open log file for writing loss curves and accuracy metrics
    std::ofstream log_file;
    log_file.open("build/" + args.log_file);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file for writing." << std::endl;
        return;
    }
    log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";

    // Calculate the number of batches
    const int input_size = model_set.train_set.num_samples;
    if (args.batch_size == 0) {
        args.batch_size = input_size; // If batch size is 0, use the entire dataset as one batch
    }
    const int num_batches = (input_size + args.batch_size - 1) / args.batch_size;

    // Create a vector of increasing indices for shuffling the training data
    std::vector<int> indices(input_size);
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < args.epochs; i++) {
        // Evaluate the model on the test set and calculate the loss and accuracy
        Matrix test_prediction = predict(model_set.test_set.features);
        Scalar test_loss = loss_func(model_set.test_set.labels, test_prediction) + args.lambda * weights_norm;
        Scalar test_acc = args.output_activation != ActivationType::LINEAR ? classification_accuracy(model_set.test_set.labels, test_prediction) * 100.0 : 0.0;

        // Shuffle the training data indices for current epoch (for stochastic gradient descent and minibatch training)
        std::shuffle(indices.begin(), indices.end(), get_random_generator());
        Matrix epoch_features = model_set.train_set.features(Eigen::placeholders::all, indices);
        Matrix epoch_labels = model_set.train_set.labels(Eigen::placeholders::all, indices);

        Scalar epoch_loss = 0.0, train_acc = 0.0;
        for (int j = 0; j < num_batches; j++) {
            // Determine the start and end indices for the current batch
            const int index_start = j * args.batch_size;
            const int index_end = std::min(index_start + args.batch_size, input_size);
            const int current_batch_size = index_end - index_start;

            // Extract the current batch of features and labels
            Matrix batch_features = epoch_features.middleCols(index_start, current_batch_size);
            Matrix batch_label = epoch_labels.middleCols(index_start, current_batch_size);

            // Forward pass to get predictions for the current batch
            Matrix batch_prediction = predict(batch_features, true);

            // Calculate the loss and accuracy for the current batch and accumulate for the epoch
            epoch_loss += loss_func(batch_label, batch_prediction) + args.lambda * weights_norm; // Include regularization term in the loss calculation
            train_acc += args.output_activation != ActivationType::LINEAR ? classification_accuracy(batch_label, batch_prediction) * 100.0 : 0.0;

            // Start the backward pass to update weights and biases based on the output loss gradient
            Matrix gradient = loss_derivative(batch_label, batch_prediction);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradient = (*it)->backward(gradient, args);
            }
        }

        // Log the average loss and accuracy for the current epoch to the log file
        log_file << i << "," << epoch_loss / num_batches << "," << test_loss << "," << train_acc / num_batches << "," << test_acc << "\n";
        log_file.flush();
    }

    log_file.close();
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(3) << std::endl
              << "==== Neural Network Training ====" << std::endl;

    Args args = parse_args(argc, argv);
    TRAIN_MODEL = args.load_file == "";

    // Set the random seed for reproducibility
    set_random_seed(args.seed);

    // Load the dataset
    ModelSet model_set = load_dataset(args.dataset_type, args.train_ratio, args.dataset_ratio);

    // Load or initialize the network
    Network* network = nullptr;
    if (!TRAIN_MODEL) {
        network = load_model(args.load_file, &args);
        if (network == nullptr) {
            std::cerr << "Failed to load the model. Exiting." << std::endl;
            return 1;
        }
    } else {
        network = new Network(args, model_set.train_set.num_features, model_set.train_set.num_classes);

        // Start training
        network->train(model_set);
    }

    // Evaluate the model
    Matrix final_train_predictions = network->predict(model_set.train_set.features);
    Matrix final_test_predictions = network->predict(model_set.test_set.features);

    // Save and cleanup
    if (args.dump_file != "" && TRAIN_MODEL) {
        dump(args.dump_file, args, network);
    }
    delete network;

    // ======================================================================================
    //                              Display final results

    if (args.output_activation != ActivationType::LINEAR) {
        std::cout << std::endl
                  << "Classification Results:" << "\n"
                  << std::left << std::setw(25) << " • Train Accuracy:" << classification_accuracy(model_set.train_set.labels, final_train_predictions) * 100.0 << "%\n"
                  << std::left << std::setw(25) << " • Test Accuracy:" << classification_accuracy(model_set.test_set.labels, final_test_predictions) * 100.0 << "%\n"
                  << std::left << std::setw(25) << " • Train Confidence:" << classification_confidence(model_set.train_set.labels, final_train_predictions) << "\n"
                  << std::left << std::setw(25) << " • Test Confidence:" << classification_confidence(model_set.test_set.labels, final_test_predictions) << "\n\n";
    }

    std::cout << "================================" << std::endl;

    std::uniform_int_distribution<int> dist(0, model_set.test_set.num_samples - 1);
    for (int i = 0; i < std::min(5, model_set.test_set.num_samples); ++i) {
        int sample_index = dist(get_random_generator());

        std::cout << "\nSample Prediction (Index: " << sample_index << ")\n";
        std::cout << "Target Label: " << model_set.test_set.labels.col(sample_index).transpose() << "\n";
        std::cout << "Prediction:   " << final_test_predictions.col(sample_index).transpose() << "\n";

        std::cout << std::endl
                  << "-------------------------------" << std::endl;
    }

    return 0;
}