#include "layer.hpp"

#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <utility>

/*
fan_in:  Number of input units
fan_out: Number of output units

RANDOM:
  Uniform distribution in [-1, 1]

LECUN:
  Uniform distribution in [-sqrt(3 / fan_in), sqrt(3 / fan_in)]

GLOROT:
  Uniform distribution in [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]

HE:
  Normal distribution with mean 0 and standard deviation sqrt(2 / fan_in)
*/
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
DenseLayer::DenseLayer(Parameters params, OptimizerType opt_type, NetworkMode mode) : params(std::move(params)) {
    if (mode == NetworkMode::TRAIN) {
        setOptimizer(opt_type);
    }
}

/*
Forward:
  net_t = sum_u(w_tu * o_u)
  o_t = f_t(net_t)

Backward:
  Etot = sum_p(Ep) = sum_p(loss(o_k, d_k))
  dEtot/dw = sum_p(dEp/dw)
  dEp/dw_tu = dEp/dnet_t * dnet_t/w_tu = dEp/dnet_t * (dsum_u(w_tu * o_u) / dw_tu)
  dEp/dw_tu (weights_delta) = delta_t * o_u

  delta_t (output_gradient) = dEp/dnet_t = dEp/do_t * do_t/dnet_t = dEp/do_t * f'_t(net_t)
  -output unit (k):
    delta_k = dEp/dnet_k = dEp/do_k * do_k/dnet_k = loss'(o_k, d_k) * f'_k(net_k) ~= (d_k - o_k) * f'_k(net_k)
  -hidden unit (j):
    delta_j = dEp/dnet_j = dEp/do_j * do_j/dnet_j = (dEp/dnet_k * dnet_k/do_j) * do_j/dnet_j = sum_k(delta_k * w_kj) * f'_j(net_j)
*/

// Forward pass through the DenseLayer, saving the input for backpropagation
Matrix DenseLayer::forward(const Matrix& input_matrix, NetworkMode mode) {
    if (params.W.cols() != input_matrix.rows()) {
        throw std::runtime_error("Dimension mismatch in DenseLayer forward pass");
    }
    if (mode == NetworkMode::TRAIN) {
        X = input_matrix; // Store the input for backpropagation
    }
    // Return the output of the layer by multiplying the weights with the input and adding the bias
    // net_t = sum_u(w_tu * o_u) + w_t0 or Y = W * X + b (for every pattern)
    return (params.W * input_matrix).colwise() + params.b;
}

// Backward pass through the DenseLayer, updating weights and biases based on the output gradient
Matrix DenseLayer::backward(const Matrix& output_gradient, const Model& model, Scalar decay_fraction, bool is_first_layer) {
    const int batch_size = static_cast<int>(X.cols());
    // Calculate the gradient updates: dE/dw_tu = (1 / mb) * sum_p(dE/dnet_t * dnet_t/dw_tu) = (1 / mb) * sum_p(delta_t * o_u)
    Matrix weights_delta = (output_gradient * X.transpose()) / batch_size; // dE/dw_tu = (1 / mb) * sum_p(dE/dnet_t * o_u)
    Vector bias_delta = output_gradient.rowwise().sum() / batch_size;      // dE/dw_t0 = (1 / mb) * sum_p(dE/dnet_t)

    // Calculate the neuron gradient to propagate to the previous layer. Skipped for the first layer of the network
    Matrix input_gradient;
    if (!is_first_layer) {
        // dE/do_u = sum_t(w_tu * dE/dnet_t) = delta_u, the sum over the units this unit connects to in the next layer
        input_gradient = params.W.transpose() * output_gradient;
    }

    // Update weights and biases using the optimizer
    optimizer->update(params, Parameters(weights_delta, bias_delta), model, decay_fraction);

    return input_gradient;
}

// ActivationLayer constructor that initializes the activation function and its derivative based on the specified ActivationType
ActivationLayer::ActivationLayer(ActivationType activation_type, bool derivative_in_loss) {
    const std::optional<ActivationPair> functions = Lookup::activation_for(activation_type);
    if (!functions) {
        throw std::invalid_argument("Unsupported activation function type");
    }
    activation.function = functions->function;
    // Left null when the derivative would do nothing or is already integrated into the loss
    const bool skip_derivative = derivative_in_loss || ActivationFunctions::has_identity_derivative(activation_type);
    activation.derivative = skip_derivative ? nullptr : functions->derivative;
}

// Forward pass through the ActivationLayer, applying the activation function to the input matrix
Matrix ActivationLayer::forward(const Matrix& input_matrix, NetworkMode mode) {
    if (mode == NetworkMode::TRAIN && activation.derivative) {
        X = input_matrix; // Only stored when backward pass will actually use it to calculate the derivative
    }
    // o_t = f_t(net_t)
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

// Forward pass through the DropoutLayer, generating a mask and applying the dropout
Matrix DropoutLayer::forward(const Matrix& input_matrix, NetworkMode mode) {
    // If not training or drop probability is 0, don't apply dropout
    if (mode == NetworkMode::TEST || probability == Scalar(0)) {
        return input_matrix;
    }

    Scalar p = Scalar(1) - probability;
    mask = Matrix(input_matrix.rows(), input_matrix.cols());
    std::uniform_real_distribution<Scalar> dist(Scalar(0), Scalar(1));
    auto& gen = get_trial_generator();

    // Generate the mask
    for (Eigen::Index i = 0; i < mask.size(); ++i) {
        // Scale the kept neurons by 1/p to maintain the expected value of the outputs
        mask(i) = (dist(gen) < p) ? (Scalar(1) / p) : Scalar(0);
    }

    // o_t = mask(net_t)
    return input_matrix.cwiseProduct(mask);
}

// Backward pass through the DropoutLayer, applying the mask to the output gradient
Matrix DropoutLayer::backward(const Matrix& output_gradient, const Model&, Scalar, bool) {
    if (probability == Scalar(0)) {
        return output_gradient;
    }

    // dE/do_u = mask(dE/dnet_t)
    return output_gradient.cwiseProduct(mask);
}