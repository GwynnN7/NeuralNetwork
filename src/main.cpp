#include "cli.hpp"
#include "dataset.hpp"
#include "functions.hpp"
#include "types.hpp"

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

class Layer {
  protected:
    Matrix X;
    Matrix Y;

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Args& args) = 0;
    virtual Scalar weight_norm() const { return 0.0; }
};

class DenseLayer : public Layer {
  private:
    Matrix W;
    Vector b;
    Matrix delta_W;

  public:
    DenseLayer(int input_size, int output_size) {
        // Initialize weights and biases pre-transposed for matrix multiplication
        Scalar limit = 1.0 / std::sqrt(static_cast<Scalar>(input_size)); // Xavier/Glorot initialization range
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        W = Matrix::NullaryExpr(output_size, input_size, [&]() { return dist(get_random_generator()); });

        b = Vector::Zero(output_size);
        delta_W = Matrix::Zero(output_size, input_size);
    }

    Matrix forward(const Matrix& input_matrix, bool training) override {
        Matrix output_matrix = (W * input_matrix).colwise() + b; // Multiply weights with input and add bias
        if (training) {
            X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
        }
        return output_matrix;
    }

    Matrix backward(const Matrix& output_gradient, const Args& args) override {
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

    Scalar weight_norm() const override { return W.squaredNorm(); }
};

class ActivationLayer : public Layer {
  private:
    std::function<Matrix(const Matrix&)> activation;
    std::function<Matrix(const Matrix&)> activation_derivative;

  public:
    ActivationLayer(ActivationType activationType) {
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

    Matrix forward(const Matrix& input_matrix, bool training) override {
        Matrix output_matrix = activation(input_matrix); // Apply the activation function to the input matrix
        if (training) {
            X = input_matrix, Y = output_matrix; // Store input and output for backpropagation
        }
        return output_matrix;
    }

    Matrix backward(const Matrix& output_gradient, [[maybe_unused]] const Args& args) override {
        Matrix derivative = activation_derivative(X);
        return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
    }
};

class Network {
  private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::function<Scalar(const Matrix&, const Matrix&)> loss_func;
    std::function<Matrix(const Matrix&, const Matrix&)> loss_derivative;
    Scalar weights_norm;

    Args args;
    std::ofstream log_file;

  public:
    Network(const Args& cli_args) {
        args = cli_args;
        log_file.open("build/" + args.log_file);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open " << args.log_file << " for writing." << std::endl;
        }
    }

    ~Network() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void setLossFunction(LossType lossType) {
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

    void addLayer(Layer* layer) { this->layers.push_back(std::unique_ptr<Layer>(layer)); }

    Matrix predict(Matrix out, bool training = false) {
        weights_norm = 0.0;
        for (auto& layer : layers) {
            out = layer->forward(out, training);
            weights_norm += layer->weight_norm();
        }
        return out;
    }

    void train(const ModelSet& model_set) {
        log_file << "epoch,train_loss,test_loss,train_acc,test_acc\n";

        const int input_size = model_set.train_set.num_samples;
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
};

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    ModelSet model_set = load_dataset(args.dataset_type, args.train_ratio, args.dataset_ratio);

    Network net(args);
    for (size_t i = 0; i < args.net_struct.size(); ++i) {
        int input_features = i == 0 ? model_set.train_set.num_features : args.net_struct[i - 1];
        int num_neurons = args.net_struct[i];
        net.addLayer(new DenseLayer(input_features, num_neurons));
        net.addLayer(new ActivationLayer(args.hidden_activation));
    }
    net.addLayer(new DenseLayer(args.net_struct.back(), model_set.train_set.num_classes));
    net.addLayer(new ActivationLayer(args.output_activation));

    if (args.output_activation == ActivationType::SOFTMAX) {
        net.setLossFunction(LossType::CCE);
    } else {
        net.setLossFunction(LossType::MSE);
    }

    net.train(model_set);

    Matrix final_train_predictions = net.predict(model_set.train_set.features);
    Matrix final_test_predictions = net.predict(model_set.test_set.features);

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

    return 0;
}