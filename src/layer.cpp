#include "layer.hpp"

#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <utility>

// DenseLayer constructor that initializes weights based on the specified initialization type
DenseLayer::DenseLayer(int input_size, int output_size, InitType init_type, OptimizerType opt_type) {
    // Determine the distribution value based on the initialization type.
    Scalar distribution_value;
    switch (init_type) {
    case InitType::RANDOM:
        distribution_value = 1.0;
        break;
    case InitType::LECUN:
        // sqrt(3 / fan_in)
        distribution_value = std::sqrt(3.0 / static_cast<Scalar>(input_size));
        break;
    case InitType::GLOROT:
        // sqrt(6 / (fan_in + fan_out))
        distribution_value = std::sqrt(6.0 / static_cast<Scalar>(input_size + output_size));
        break;
    case InitType::HE:
        // Normal with sigma = sqrt(2 / fan_in)
        distribution_value = std::sqrt(2.0 / static_cast<Scalar>(input_size));
        break;
    default:
        throw std::invalid_argument("Unsupported initialization type");
    }

    // Initialize the weights and biases of the layer using the determined distribution value
    params = Parameters(output_size, input_size);
    switch (init_type) {
    case InitType::HE: {
        std::normal_distribution<Scalar> normal_dist(0.0, distribution_value);
        for (Eigen::Index i = 0; i < params.W.size(); ++i) {
            params.W(i) = normal_dist(get_trial_generator());
        }
    } break;
    default: {
        std::uniform_real_distribution<Scalar> uniform_dist(-distribution_value, distribution_value);
        for (Eigen::Index i = 0; i < params.W.size(); ++i) {
            params.W(i) = uniform_dist(get_trial_generator());
        }
    } break;
    }

    setOptimizer(opt_type);
}

// DenseLayer constructor that initializes weights and biases with the provided matrices and vectors
DenseLayer::DenseLayer(Parameters params, OptimizerType opt_type, bool instantiate_optimizer) : params(std::move(params)) {
    if (instantiate_optimizer) {
        setOptimizer(opt_type);
    }
}

// Forward pass through the DenseLayer, saving the input for backpropagation
Matrix DenseLayer::forward(const Matrix& input_matrix, bool training) {
    if (params.W.cols() != input_matrix.rows()) {
        throw std::runtime_error("Dimension mismatch in DenseLayer forward pass");
    }
    if (training) {
        X = input_matrix; // Store the input for backpropagation
    }
    // Return the output of the layer by multiplying the weights with the input and adding the bias
    // net_t = sum_u(w_tu * o_u) + w_t0, for every pattern; or Y = W * X + b
    return (params.W * input_matrix).colwise() + params.b;
}

// Backward pass through the DenseLayer, updating weights and biases based on the output gradient
Matrix DenseLayer::backward(const Matrix& output_gradient, const Model& model, Scalar batch_fraction, bool is_first_layer) {
    const int batch_size = static_cast<int>(X.cols());
    // Calculate the gradients for weights and biases
    Matrix weights_delta = (output_gradient * X.transpose()) / batch_size; // dE/dw_tu = (1 / mb) * sum_p(dE/dnet_t * o_u)
    Vector bias_delta = output_gradient.rowwise().sum() / batch_size;      // dE/dw_t0 = (1 / mb) * sum_p(dE/dnet_t)

    // Calculate the neuron gradient to propagate to the previous layer. Skipped for the first layer of the network
    Matrix input_gradient;
    if (!is_first_layer) {
        // dE/do_u = sum_t(w_tu * dE/dnet_t), the sum over the units this unit connects to in the next layer
        input_gradient = params.W.transpose() * output_gradient;
    }

    // Update weights and biases using the optimizer
    optimizer->update(params, Parameters(weights_delta, bias_delta), model, batch_fraction);

    return input_gradient;
}

// ActivationLayer constructor that initializes the activation function and its derivative based on the specified ActivationType
ActivationLayer::ActivationLayer(ActivationType activation_type, bool derivative_in_loss) {
    const std::optional<ActivationPair> functions = Lookup::activation_for(activation_type);
    if (!functions) {
        throw std::invalid_argument("Unsupported activation function type");
    }
    activation.function = functions->function;
    // Left null when the derivative would do nothing or is already folded into the loss
    const bool skip_derivative = derivative_in_loss || ActivationFunctions::has_identity_derivative(activation_type);
    activation.derivative = skip_derivative ? nullptr : functions->derivative;
}

// Forward pass through the ActivationLayer, applying the activation function to the input matrix
Matrix ActivationLayer::forward(const Matrix& input_matrix, bool training) {
    if (training && activation.derivative) {
        X = input_matrix; // Only stored when backward pass will actually use it to calculate the derivative
    }
    // o_t = f(net_t)
    return activation.function(input_matrix); // Call the activation function on the input matrix
}

// Backward pass through the ActivationLayer, calculating the gradient with respect to the input
Matrix ActivationLayer::backward(const Matrix& output_gradient, const Model&, Scalar, bool) {
    if (!activation.derivative) {
        return output_gradient; // Multiplying by this derivative would do nothing or fail
    }
    // dE/dnet_t = dE/do_t * f'(net_t)
    Matrix derivative = activation.derivative(X);    // Calculate the derivative of the activation function with respect to the input
    return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
}