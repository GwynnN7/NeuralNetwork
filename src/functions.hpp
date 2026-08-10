#pragma once

#include "types.hpp"

// Activation Functions

// Activation Functions: o_t = f(net_t), where net_t = sum_u(w_tu * o_u) is the net input of a unit
namespace ActivationFunctions {
inline Matrix sigmoid(const Matrix& net) {
    // f(net) = 1 / (1 + e^(-net))
    return Scalar(1) / (Scalar(1) + (-net.array()).exp());
}

inline Matrix sigmoid_derivative(const Matrix& net) {
    // f'(net) = f(net) * (1 - f(net))
    Matrix s = sigmoid(net);
    return s.array() * (Scalar(1) - s.array());
}

inline Matrix relu(const Matrix& net) {
    // f(net) = max(0, net)
    return net.array().max(Scalar(0));
}

inline Matrix relu_derivative(const Matrix& net) {
    // f'(net) = 1 if net > 0, else 0
    return (net.array() > Scalar(0)).template cast<Scalar>();
}

inline Matrix tanh_activation(const Matrix& net) {
    // f(net) = (e^net - e^(-net)) / (e^net + e^(-net)) = tanh(net)
    return net.array().tanh();
}

inline Matrix tanh_derivative(const Matrix& net) {
    // f'(net) = 1 - f(net)^2
    Matrix t = tanh_activation(net);
    return Scalar(1) - (t.array() * t.array());
}

inline Matrix linear(const Matrix& net) {
    // f(net) = net
    return net;
}

inline Matrix linear_derivative(const Matrix& net) {
    // f'(net) = 1 (see has_identity_derivative)
    return Matrix::Ones(net.rows(), net.cols());
}

inline Matrix softmax(const Matrix& net) {
    // f(net_k) = e^(net_k) / sum_j(e^(net_j))
    Matrix inputs = net.rowwise() - net.colwise().maxCoeff(); // Avoid e^x overflow
    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array().rowwise() / sum.row(0).array();
}

inline Matrix softmax_derivative(const Matrix&) {
    // Softmax derivative is handled in the loss function (CCE), so this function is never called
    throw std::logic_error("Softmax is only supported as an output activation paired with CCE loss");
}

// Used to skip the derivative in backward pass when it would do nothing
constexpr bool has_identity_derivative(ActivationType activation) noexcept {
    return activation == ActivationType::LINEAR;
}
} // namespace ActivationFunctions

// Functions to compute loss and its derivative (error, without regularization)
namespace LossFunctions {
inline Scalar mse(const Matrix& target, const Matrix& prediction) {
    // E = (1 / l) * sum_p(sum_k((d_k - o_k)^2))
    return (prediction - target).colwise().squaredNorm().mean();
}

inline Scalar mee(const Matrix& target, const Matrix& prediction) {
    // E = (1 / l) * sum_p(sqrt(sum_k((d_k - o_k)^2)))
    return (prediction - target).colwise().norm().mean();
}

inline Matrix mse_derivative(const Matrix& target, const Matrix& prediction) {
    // dE/do_k = 2 * (o_k - d_k)
    return 2 * (prediction - target);
}

inline Matrix mee_derivative(const Matrix& target, const Matrix& prediction) {
    // dE/do_k = (o_k - d_k) / sqrt(sum_k((o_k - d_k)^2))
    const Matrix numerator = prediction - target;
    const Matrix denominator = numerator.colwise().norm().cwiseMax(LOSS_EPSILON);
    return numerator.array().rowwise() / denominator.array().row(0);
}

inline Scalar bce(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(LOSS_EPSILON).cwiseMin(Scalar(1) - LOSS_EPSILON);
    // E = -(1 / l) * sum_p(sum_k(d_k * log(o_k) + (1 - d_k) * log(1 - o_k)))
    return -(target.array() * pred_clipped.array().log() +
             (Scalar(1) - target.array()) * (Scalar(1) - pred_clipped.array()).log())
                .sum() /
           target.cols();
}

inline Matrix bce_derivative(const Matrix& target, const Matrix& prediction) {
    // Combined derivative of BCE + Sigmoid: dE/dnet_k = o_k - d_k (see includes_output_derivative)
    return (prediction - target);
}

inline Scalar cce(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(LOSS_EPSILON).cwiseMin(Scalar(1) - LOSS_EPSILON);
    // E = -(1 / l) * sum_p(sum_k(d_k * log(o_k)))
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
    // Combined derivative of CCE + Softmax: dE/dnet_k = o_k - d_k (see includes_output_derivative)
    return (prediction - target);
}

// True when the loss derivative already simplifies the output activation's derivative
constexpr bool includes_output_derivative(LossType loss) noexcept {
    return loss == LossType::BCE || loss == LossType::CCE;
}
} // namespace LossFunctions

// Lookup tables for activation and loss functions
namespace Lookup {
inline constexpr std::array<std::pair<ActivationType, ActivationPair>, 5> activation_functions{{
    {ActivationType::SIGMOID, {ActivationFunctions::sigmoid, ActivationFunctions::sigmoid_derivative}},
    {ActivationType::RELU, {ActivationFunctions::relu, ActivationFunctions::relu_derivative}},
    {ActivationType::TANH, {ActivationFunctions::tanh_activation, ActivationFunctions::tanh_derivative}},
    {ActivationType::LINEAR, {ActivationFunctions::linear, ActivationFunctions::linear_derivative}},
    {ActivationType::SOFTMAX, {ActivationFunctions::softmax, ActivationFunctions::softmax_derivative}},
}};

inline constexpr std::array<std::pair<LossType, LossPair>, 4> loss_functions{{
    {LossType::MSE, {LossFunctions::mse, LossFunctions::mse_derivative}},
    {LossType::MEE, {LossFunctions::mee, LossFunctions::mee_derivative}},
    {LossType::BCE, {LossFunctions::bce, LossFunctions::bce_derivative}},
    {LossType::CCE, {LossFunctions::cce, LossFunctions::cce_derivative}},
}};

constexpr std::optional<ActivationPair> activation_for(ActivationType type) noexcept {
    const auto entry = std::ranges::find(activation_functions, type, &std::pair<ActivationType, ActivationPair>::first);
    return entry != activation_functions.end() ? std::optional(entry->second) : std::nullopt;
}

constexpr std::optional<LossPair> loss_for(LossType type) noexcept {
    const auto entry = std::ranges::find(loss_functions, type, &std::pair<LossType, LossPair>::first);
    return entry != loss_functions.end() ? std::optional(entry->second) : std::nullopt;
}

} // namespace Lookup