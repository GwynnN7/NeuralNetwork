#include "cli.hpp"
#include "dataset.hpp"
#include "dump.hpp"
#include "functions.hpp"
#include "network.hpp"
#include "types.hpp"

#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

bool TRAINING = true;

// DenseLayer class implementation

DenseLayer::DenseLayer(int input_size, int output_size, InitializationType init_type) {
    // Initialize weights and biases pre-transposed for matrix multiplication
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

Matrix DenseLayer::forward(const Matrix& input_matrix, bool training) {
    if (W.cols() != input_matrix.rows()) {
        throw std::runtime_error("Dimension mismatch: Layer expects " + std::to_string(W.cols()) + " features, but received " + std::to_string(input_matrix.rows()));
    }
    Matrix output_matrix = (W * input_matrix).colwise() + b; // Multiply weights with input and add bias
    if (training) {
        X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
    }
    return output_matrix;
}

Matrix DenseLayer::backward(const Matrix& output_gradient, const Args& args) {
    Matrix weights_delta = (output_gradient * X.transpose()) / args.batch_size; // Average over batch size for gradient
    Vector bias_delta = output_gradient.rowwise().sum() / args.batch_size;      // Sum over columns to aggregate
    // batch gradients and average
    Matrix input_gradient = W.transpose() * output_gradient;

    delta_W = -args.eta * weights_delta + args.alpha * delta_W; // Learning rate and Momentum update
    Matrix l2_penalty = args.eta * args.lambda * W;             // L2 regularization term

    W = W + delta_W - l2_penalty;
    b -= args.eta * bias_delta;

    return input_gradient;
}

// ActivationLayer class implementation

ActivationLayer::ActivationLayer(ActivationType activationType) {
    switch (activationType) {
    case ActivationType::RELU:
        activation = relu;
        activation_derivative = relu_derivative;
        break;
    case ActivationType::SIGMOID:
        activation = sigmoid;
        activation_derivative = sigmoid_derivative;
        break;
    case ActivationType::TANH:
        activation = tanh_activation;
        activation_derivative = tanh_derivative;
        break;
    case ActivationType::SOFTMAX:
        activation = softmax;
        activation_derivative = softmax_derivative;
        break;
    case ActivationType::LINEAR:
        activation = linear;
        activation_derivative = linear_derivative;
        break;
    default:
        throw std::invalid_argument("Unsupported activation function type.");
        break;
    }
}

Matrix ActivationLayer::forward(const Matrix& input_matrix, bool training) {
    Matrix output_matrix = activation(input_matrix); // Apply the activation function to the input matrix
    if (training) {
        X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
    }
    return output_matrix;
}

Matrix ActivationLayer::backward(const Matrix& output_gradient, [[maybe_unused]] const Args& args) {
    Matrix derivative = activation_derivative(X);
    return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
}

// Network class implementation

Network::Network(const Args& cli_args) {
    args = cli_args;

    if (TRAINING) {
        log_file.open("build/" + args.log_file);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open " << args.log_file << " for writing." << std::endl;
        }
    }

    if (args.output_activation == ActivationType::SOFTMAX) {
        setLossFunction(LossType::CCE);
    } else {
        setLossFunction(LossType::MSE);
    }
}

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

Network::Network(const Args& cli_args, std::vector<Matrix> weights, std::vector<Vector> biases) : Network(cli_args) {
    for (size_t i = 0; i < args.net_struct.size(); ++i) {
        addLayer(new DenseLayer(weights[i], biases[i]));
        addLayer(new ActivationLayer(args.hidden_activation));
    }
    addLayer(new DenseLayer(weights.back(), biases.back()));
    addLayer(new ActivationLayer(args.output_activation));
}

Network::~Network() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

void Network::setLossFunction(LossType lossType) {
    switch (lossType) {
    case LossType::MSE:
        loss_func = mse;
        loss_derivative = mse_derivative;
        break;
    case LossType::CCE:
        loss_func = cce;
        loss_derivative = cce_derivative;
        break;
    default:
        throw std::invalid_argument("Unsupported loss function type.");
        break;
    }
}

void Network::addLayer(Layer* layer) {
    this->layers.push_back(std::unique_ptr<Layer>(layer));
}

std::vector<const DenseLayer*> Network::getDenseLayers() const {
    std::vector<const DenseLayer*> dense_layers;
    for (const auto& layer : layers) {
        if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
            dense_layers.push_back(dense_layer);
        }
    }
    return dense_layers;
}

Matrix Network::predict(Matrix out, bool training) {
    weights_norm = 0.0;
    for (auto& layer : layers) {
        out = layer->forward(out, training);
        weights_norm += layer->weight_norm();
    }
    return out;
}

void Network::train(const ModelSet& model_set) {
    std::cout << "\nTraining Configuration:" << "\n"
              << std::left << std::setw(25) << " • Epochs:" << args.epochs << "\n"
              << std::left << std::setw(25) << " • Batch Size:" << args.batch_size << "\n"
              << std::left << std::setw(25) << " • Learning Rate:" << args.eta << "\n"
              << std::left << std::setw(25) << " • Regularization:" << args.lambda << "\n"
              << std::left << std::setw(25) << " • Momentum:" << args.alpha << "\n\n";

    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file for writing." << std::endl;
        return;
    }
    log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";

    const int input_size = model_set.train_set.num_samples;
    if (args.batch_size == 0) {
        args.batch_size = input_size; // If batch size is 0, use the entire dataset as one batch
    }
    const int num_batches = (input_size + args.batch_size - 1) / args.batch_size;

    std::vector<int> indices(input_size);
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < args.epochs; i++) {

        Matrix test_prediction = predict(model_set.test_set.features);
        Scalar test_loss = loss_func(model_set.test_set.labels, test_prediction) + args.lambda * weights_norm;
        Scalar test_acc = args.output_activation != ActivationType::LINEAR ? classification_accuracy(model_set.test_set.labels, test_prediction) * 100.0 : 0.0;

        std::shuffle(indices.begin(), indices.end(), get_random_generator());
        Matrix epoch_features = model_set.train_set.features(Eigen::placeholders::all, indices);
        Matrix epoch_labels = model_set.train_set.labels(Eigen::placeholders::all, indices);

        Scalar epoch_loss = 0.0;
        Scalar train_acc = 0.0;
        for (int j = 0; j < num_batches; j++) {
            const int index_start = j * args.batch_size;
            const int index_end = std::min(index_start + args.batch_size, input_size);
            const int current_batch_size = index_end - index_start;

            Matrix batch_features = epoch_features.middleCols(index_start, current_batch_size);
            Matrix batch_label = epoch_labels.middleCols(index_start, current_batch_size);

            Matrix batch_prediction = predict(batch_features, true);

            epoch_loss += loss_func(batch_label, batch_prediction) + args.lambda * weights_norm;
            train_acc += args.output_activation != ActivationType::LINEAR ? classification_accuracy(batch_label, batch_prediction) * 100.0 : 0.0;

            Matrix gradient = loss_derivative(batch_label, batch_prediction);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradient = (*it)->backward(gradient, args);
            }
        }

        log_file << i << "," << epoch_loss / num_batches << "," << test_loss << "," << train_acc / num_batches << "," << test_acc << "\n";
        log_file.flush();
    }

    log_file.close();
}

int main(int argc, char* argv[]) {
    std::cout << std::endl
              << "== Neural Network Training ==" << std::endl;
    Args args = parse_args(argc, argv);
    TRAINING = args.load_file == "";

    Network* network = nullptr;
    if (!TRAINING) {
        std::cout << "Loading model from: " << args.load_file << std::endl;
        network = load_model(args.load_file, &args);
    }

    ModelSet model_set = load_dataset(args.dataset_type, args.train_ratio, args.dataset_ratio);

    if (TRAINING) {
        network = new Network(args, model_set.train_set.num_features, model_set.train_set.num_classes);
        network->train(model_set);
    } else {
        if (network == nullptr) {
            std::cerr << "Failed to load model from: " << args.load_file << std::endl;
            return 1;
        }
        std::cout << "Model loaded successfully." << std::endl;
    }

    Matrix final_train_predictions = network->predict(model_set.train_set.features);
    Matrix final_test_predictions = network->predict(model_set.test_set.features);

    if (args.output_activation != ActivationType::LINEAR) {
        std::cout << "Final Train Accuracy: "
                  << classification_accuracy(model_set.train_set.labels, final_train_predictions) * 100.0 << "%\n";
        std::cout << "Final Test Accuracy: "
                  << classification_accuracy(model_set.test_set.labels, final_test_predictions) * 100.0 << "%\n";
    }

    std::uniform_int_distribution<int> dist(0, model_set.test_set.num_samples - 1);
    int sample_index = dist(get_random_generator());

    std::cout << "\nSample Prediction (Test Set Index " << sample_index << ")\n";

    std::cout << "Target Label: "
              << model_set.test_set.labels.col(sample_index).transpose() << "\n";
    std::cout << "Prediction:   "
              << std::fixed << std::setprecision(4)
              << final_test_predictions.col(sample_index).transpose() << "\n";

    if (args.dump_file != "") {
        std::cout << "Saving model to: " << args.dump_file << std::endl;
        dump(args.dump_file, args, network);
    }
    delete network;

    return 0;
}