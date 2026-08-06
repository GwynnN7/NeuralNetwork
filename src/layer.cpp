#include "layer.hpp"

#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <random>

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

    // Initialize weights and biases pre-transposed for more efficient matrix multiplication later on
    W = Matrix::Zero(output_size, input_size);
    switch (init_type) {
    case InitType::HE: {
        std::normal_distribution<Scalar> normal_dist(0.0, distribution_value);
        for (Eigen::Index i = 0; i < W.size(); ++i) {
            W(i) = normal_dist(get_random_generator());
        }
    } break;
    default: {
        std::uniform_real_distribution<Scalar> uniform_dist(-distribution_value, distribution_value);
        for (Eigen::Index i = 0; i < W.size(); ++i) {
            W(i) = uniform_dist(get_random_generator());
        }
    } break;
    }

    setOptimizer(opt_type);
    b = Vector::Zero(output_size);
}

// DenseLayer constructor that initializes weights and biases with the provided matrices and vectors
DenseLayer::DenseLayer(Matrix weights, Vector biases, OptimizerType opt_type, bool instantiate_optimizer) : W(std::move(weights)), b(std::move(biases)) {
    if (instantiate_optimizer) {
        setOptimizer(opt_type);
    }
}

// Forward pass through the DenseLayer, saving the input for backpropagation
Matrix DenseLayer::forward(const Matrix& input_matrix, bool training) {
    if (W.cols() != input_matrix.rows()) {
        throw std::runtime_error("Dimension mismatch in DenseLayer forward pass");
    }
    if (training) {
        X = input_matrix; // Store the input for backpropagation
    }
    return (W * input_matrix).colwise() + b; // Multiply weights with input and add bias
}

// Backward pass through the DenseLayer, updating weights and biases based on the output gradient
Matrix DenseLayer::backward(const Matrix& output_gradient, const Model& model, bool is_first_layer) {
    Matrix weights_delta = (output_gradient * X.transpose()) / X.cols(); // Calculate the delta of weights (average over the batch)
    Vector bias_delta = output_gradient.rowwise().sum() / X.cols();      // Calculate the delta of biases (sum over columns to aggregate, and average over the batch)

    // Calculate the neuron gradient to propagate to the previous layer. Skipped for the first layer of the network
    Matrix input_gradient;
    if (!is_first_layer) {
        input_gradient = W.transpose() * output_gradient;
    }

    // Update weights and biases using the optimizer
    optimizer->update(W, b, weights_delta, bias_delta, model);

    return input_gradient;
}

// ActivationLayer constructor that initializes the activation function and its derivative based on the specified ActivationType
ActivationLayer::ActivationLayer(ActivationType activation_type, bool derivative_in_loss) {
    try {
        activation = Maps::activation_map.at(activation_type).first;
        activation_derivative = Maps::activation_map.at(activation_type).second;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("Unsupported activation function type");
    }
    skip_derivative = derivative_in_loss || ActivationFunctions::has_identity_derivative(activation_type);
}

// Forward pass through the ActivationLayer, applying the activation function to the input matrix
Matrix ActivationLayer::forward(const Matrix& input_matrix, bool training) {
    if (training && !skip_derivative) {
        X = input_matrix; // Only stored when backward pass will actually use it to calculate the derivative
    }
    return activation(input_matrix); // Call the activation function on the input matrix
}

// Backward pass through the ActivationLayer, calculating the gradient with respect to the input
Matrix ActivationLayer::backward(const Matrix& output_gradient, const Model&, bool) {
    if (skip_derivative) {
        return output_gradient; // Multiplying by this derivative would do nothing or fail
    }
    Matrix derivative = activation_derivative(X);    // Calculate the derivative of the activation function with respect to the input
    return output_gradient.cwiseProduct(derivative); // Element-wise multiplication of gradient and derivative
}