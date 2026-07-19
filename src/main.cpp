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
    virtual Matrix forward(const Matrix& input_matrix) = 0;
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
        W = Matrix::Random(output_size, input_size);
        b = Vector::Zero(output_size);
        delta_W = Matrix::Zero(output_size, input_size);
    }

    Matrix forward(const Matrix& input_matrix) override {
        X = input_matrix;
        Y = (W * X).colwise() + b; // Multiply weights with input and add bias
        return Y;
    }

    Matrix backward(const Matrix& output_gradient, const Args& args) override {
        Matrix weights_delta = (output_gradient * X.transpose()) / args.batch_size; // Average over batch size for gradient
        Vector bias_delta = output_gradient.rowwise().sum() / args.batch_size;      // Sum over columns to aggregate
        // batch gradients and average
        Matrix input_gradient = W.transpose() * output_gradient;

        delta_W = -args.eta * weights_delta + args.alpha * delta_W; // Learning rate and Momentum update
        Matrix l2_penalty = args.lambda * W;                        // L2 regularization term

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

    Matrix forward(const Matrix& input_matrix) override {
        X = input_matrix;
        Y = activation(X); // Apply the activation function to the input matrix
        return Y;
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

    Matrix predict(Matrix out) {
        weights_norm = 0.0;
        for (auto& layer : layers) {
            out = layer->forward(out);
            weights_norm += layer->weight_norm();
        }
        return out;
    }

    void train(Matrix input, Matrix target) {
        log_file << "epoch,train_loss,test_loss\n";

        const int input_size = input.cols();
        const int num_batches = (input_size + args.batch_size - 1) / args.batch_size;

        for (int i = 0; i < args.epochs; i++) {
            Scalar batch_loss = 0.0;
            for (int j = 0; j < num_batches; j++) {
                const int index_start = j * args.batch_size;
                const int index_end = std::min(index_start + args.batch_size, input_size);
                const int current_batch_size = index_end - index_start;

                Matrix batch_input = input.middleCols(index_start, current_batch_size);
                Matrix batch_target = target.middleCols(index_start, current_batch_size);

                Matrix batch_prediction = predict(batch_input);

                batch_loss += loss_func(batch_target, batch_prediction) + args.lambda * weights_norm;

                Matrix gradient = loss_derivative(batch_target, batch_prediction);
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    gradient = (*it)->backward(gradient, args);
                }
            }
            log_file << i << "," << batch_loss / num_batches << "," << 0.0 << "\n"; // Test loss is set to 0 for now, work in progress
            log_file.flush();
        }

        log_file.close();
    }
};

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    Dataset dataset = load_dataset(args.dataset_type);

    Network net(args);
    for (size_t i = 0; i < args.net_struct.size(); ++i) {
        int input_features = i == 0 ? dataset.num_features : args.net_struct[i - 1];
        int num_neurons = args.net_struct[i];
        net.addLayer(new DenseLayer(input_features, num_neurons));
        net.addLayer(new ActivationLayer(args.hidden_activation));
    }
    net.addLayer(new DenseLayer(args.net_struct.back(), dataset.num_classes));
    net.addLayer(new ActivationLayer(args.output_activation));
    if (args.output_activation == ActivationType::SOFTMAX) {
        net.setLossFunction(LossType::CCE);
    } else {
        net.setLossFunction(LossType::MSE);
    }

    net.train(dataset.features, dataset.labels);

    Matrix predictions = net.predict(dataset.features);

    std::cout << "Final Predictions:" << "\n"
              << std::fixed << std::setprecision(4) << predictions << "\n";

    return 0;
}