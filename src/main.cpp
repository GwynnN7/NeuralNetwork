#include "cli.hpp"
#include "dataset.hpp"
#include "dump.hpp"
#include "functions.hpp"
#include "network.hpp"
#include "types.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <vector>

bool TRAIN_MODEL = true;
bool IS_REGRESSION = false;
std::string MODEL_PATH;

volatile std::sig_atomic_t early_stop_flag = 0;
void handle_sigint(int sig) {
    if (early_stop_flag) {
        std::println(stderr, "\n[Force Quit]\n");
        std::exit(sig);
    }
    early_stop_flag = 1;
}

// -- DenseLayer class implementation --

// DenseLayer constructor that initializes weights based on the specified initialization type
DenseLayer::DenseLayer(int input_size, int output_size, InitType init_type) {
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

    // Initialize weights and biases pre-transposed for efficient matrix multiplication
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
        throw std::invalid_argument("Unsupported activation function type");
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
        throw std::invalid_argument("Unsupported loss function type");
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
void Network::train(const Dataset& dataset, const SetIndices& indices, int fold_index) {
    if (fold_index == 0) {
        print_args(args);
    }

    // Open log file for writing loss curves and accuracy metrics
    std::ofstream log_file;
    log_file.open(MODEL_PATH + "/" + std::to_string(fold_index) + "_log.csv");
    if (!log_file.is_open()) {
        std::println(stderr, "Failed to open log file for writing");
        return;
    }
    log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";

    // Extract test features and labels from the dataset
    Matrix test_features = dataset.features(Eigen::placeholders::all, indices.test_indices);
    Matrix test_labels = dataset.labels(Eigen::placeholders::all, indices.test_indices);

    // Calculate the number of batches
    const int input_size = indices.train_indices.size();
    if (args.batch_size == 0) {
        args.batch_size = input_size; // If batch size is 0, use the entire dataset as one batch
    }
    const int num_batches = (input_size + args.batch_size - 1) / args.batch_size;

    // Create a local copy of the training indices to shuffle for each epoch
    std::vector<int> epoch_indices = indices.train_indices;

    for (int i = 0; i < args.epochs && !early_stop_flag; i++) {
        // Evaluate the model on the test set and calculate the loss and accuracy
        Matrix test_prediction = predict(test_features);
        Scalar test_loss = loss_func(test_labels, test_prediction) + args.lambda * weights_norm;
        Scalar test_acc = !IS_REGRESSION ? classification_accuracy(test_labels, test_prediction) * 100.0 : 0.0;

        // Shuffle the training data indices for current epoch (for stochastic gradient descent and minibatch training)
        std::ranges::shuffle(epoch_indices, get_random_generator());

        Scalar epoch_loss = 0.0, train_acc = 0.0;
        for (int j = 0; j < num_batches && !early_stop_flag; j++) {
            // Determine the start and end indices for the current batch
            const int index_start = j * args.batch_size;
            const int index_end = std::min(index_start + args.batch_size, input_size);

            // Extract the current batch of features and labels
            std::vector<int> batch_indices(epoch_indices.begin() + index_start, epoch_indices.begin() + index_end);
            Matrix batch_features = dataset.features(Eigen::placeholders::all, batch_indices);
            Matrix batch_labels = dataset.labels(Eigen::placeholders::all, batch_indices);

            // Forward pass to get predictions for the current batch
            Matrix batch_prediction = predict(batch_features, true);

            // Calculate the loss and accuracy for the current batch and accumulate for the epoch
            epoch_loss += loss_func(batch_labels, batch_prediction) + args.lambda * weights_norm; // Include regularization term in the loss calculation
            train_acc += !IS_REGRESSION ? classification_accuracy(batch_labels, batch_prediction) * 100.0 : 0.0;

            // Start the backward pass to update weights and biases based on the output loss gradient
            Matrix gradient = loss_derivative(batch_labels, batch_prediction);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradient = (*it)->backward(gradient, args);
            }
        }

        if (early_stop_flag) {
            std::println(stderr, "\n[Manual Early Stopping: Fold {}]", fold_index + 1);
            break;
        }

        // Log the average loss and accuracy for the current epoch to the log file
        log_file << i << "," << epoch_loss / num_batches << "," << test_loss << "," << train_acc / num_batches << "," << test_acc << "\n";
        log_file.flush();
    }

    log_file.close();
    early_stop_flag = 0; // Reset the early stop flag for the next fold
}

int main(int argc, char* argv[]) {
    std::println("\n\n[Neural Network Training]");

    std::signal(SIGINT, handle_sigint);
    Args args = parse_args(argc, argv);
    TRAIN_MODEL = !args.load;

    MODEL_PATH = "artifacts/" + args.name;
    std::filesystem::create_directories(MODEL_PATH);

    // Set the random seed for reproducibility
    set_random_seed(args.seed);

    // Load the dataset
    Dataset dataset = load_dataset(args.dataset_type, args.dataset_ratio);

    std::vector<SetIndices> folds = split_dataset(dataset.num_samples, args.k_folds, args.train_ratio, args.shuffle);

    // Aggregate performance variables
    MetricsResult train_metrics, test_metrics;

    for (size_t i = 0; i < folds.size(); ++i) {
        // Load or initialize the network
        Network* network = nullptr;
        if (!TRAIN_MODEL) {
            network = load_model(MODEL_PATH, &args);
            if (network == nullptr) {
                std::println(stderr, "Failed to load the model. Exiting");
                return 1;
            }
        } else {
            network = new Network(args, dataset.num_features, dataset.num_classes);
        }

        IS_REGRESSION = args.output_activation == ActivationType::LINEAR;

        if (TRAIN_MODEL) {
            network->train(dataset, folds[i], i);
        }

        // Evaluate the model
        Matrix train_features = dataset.features(Eigen::placeholders::all, folds[i].train_indices);
        Matrix train_labels = dataset.labels(Eigen::placeholders::all, folds[i].train_indices);
        Matrix test_features = dataset.features(Eigen::placeholders::all, folds[i].test_indices);
        Matrix test_labels = dataset.labels(Eigen::placeholders::all, folds[i].test_indices);

        Matrix final_train_predictions = network->predict(train_features);
        Matrix final_test_predictions = network->predict(test_features);

        train_metrics.update(train_labels, final_train_predictions, IS_REGRESSION);
        test_metrics.update(test_labels, final_test_predictions, IS_REGRESSION);

        // Save and cleanup
        if (args.dump && TRAIN_MODEL && i == folds.size() - 1) {
            dump(MODEL_PATH, args, network);
        }

        if (i == folds.size() - 1) {
            std::println("\nRandom Samples Predictions:");

            std::uniform_int_distribution<int> dist(0, folds[i].test_indices.size() - 1);
            for (int j = 0; j < std::min(3, (int)folds[i].test_indices.size()); ++j) {
                int sample_index = dist(get_random_generator());
                if (j > 0) {
                    std::println("   --------------{}", std::string(test_labels.rows() * 6 - 1, '-'));
                }
                std::cout << std::fixed << std::setprecision(3);
                std::cout << " • Target Label: " << test_labels.col(sample_index).transpose() << "\n";
                std::cout << " • Prediction:   " << final_test_predictions.col(sample_index).transpose() << "\n";
            }
        }

        delete network;
    }

    train_metrics.average(folds.size());
    test_metrics.average(folds.size());

    std::println("\nTraining Set Metrics:");
    train_metrics.print(IS_REGRESSION);
    std::println("Test Set Metrics:");
    test_metrics.print(IS_REGRESSION);
    return 0;
}