#pragma once

#include "types.hpp"

// Activation Functions

namespace ActivationFunctions {
inline Matrix sigmoid(const Matrix& X) {
    // f(x) = 1 / (1 + e^(-x))
    return Scalar(1) / (Scalar(1) + (-X.array()).exp());
}

inline Matrix sigmoid_derivative(const Matrix& X) {
    // f'(x) = f(x) * (1 - f(x))
    Matrix s = sigmoid(X);
    return s.array() * (Scalar(1) - s.array());
}

inline Matrix relu(const Matrix& X) {
    // f(x) = max(0, x)
    return X.array().max(Scalar(0));
}

inline Matrix relu_derivative(const Matrix& X) {
    // f'(x) = 1 if x > 0, else 0
    return (X.array() > Scalar(0)).template cast<Scalar>();
}

inline Matrix tanh_activation(const Matrix& X) {
    // f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = tanh(x)
    return X.array().tanh();
}

inline Matrix tanh_derivative(const Matrix& X) {
    // f'(x) = 1 - f(x)^2
    Matrix t = tanh_activation(X);
    return Scalar(1) - (t.array() * t.array());
}

inline Matrix linear(const Matrix& X) {
    // f(x) = x
    return X;
}

inline Matrix linear_derivative(const Matrix& X) {
    // f'(x) = 1 (see has_identity_derivative)
    return Matrix::Ones(X.rows(), X.cols());
}

inline Matrix softmax(const Matrix& X) {
    // f(x) = e^(x_i) / sum(e^(x_j))
    Matrix inputs = X.rowwise() - X.colwise().maxCoeff(); // Avoid e^x overflow
    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array().rowwise() / sum.row(0).array();
}

inline Matrix softmax_derivative(const Matrix&) {
    // Softmax derivative is handled in the loss function (CCE), so this function is never called
    throw std::logic_error("Softmax is only supported as an output activation paired with CCE loss");
}

// Used to skip the derivative in backward pass when it would do nothing
inline bool has_identity_derivative(ActivationType activation) {
    return activation == ActivationType::LINEAR;
}
} // namespace ActivationFunctions

namespace LossFunctions {
inline Scalar mse(const Matrix& target, const Matrix& prediction) {
    // f = (1 / N) * sum((y* - y)^2)
    return (prediction - target).array().square().colwise().sum().mean();
}

inline Matrix mse_derivative(const Matrix& target, const Matrix& prediction) {
    // f' = 2 * (y* - y)
    return 2 * (prediction - target);
}

inline Scalar bce(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(LOSS_EPS).cwiseMin(Scalar(1) - LOSS_EPS);
    // f = -(1 / N) * sum(y * log(y*) + (1 - y) * log(1 - y*))
    return -(target.array() * pred_clipped.array().log() +
             (Scalar(1) - target.array()) * (Scalar(1) - pred_clipped.array()).log())
                .sum() /
           target.cols();
}

inline Matrix bce_derivative(const Matrix& target, const Matrix& prediction) {
    // Combined derivative of BCE + Sigmoid: f' = y* - y (see includes_output_derivative)
    return (prediction - target);
}

inline Scalar cce(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(LOSS_EPS).cwiseMin(Scalar(1) - LOSS_EPS);
    // f = -(1 / N) * sum(y * log(y*))
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
    // Combined derivative of CCE + Softmax: f' = y* - y (see includes_output_derivative)
    return (prediction - target);
}

// True when the loss derivative already simplifies the output activation's derivative
inline bool includes_output_derivative(LossType loss) {
    return loss == LossType::BCE || loss == LossType::CCE;
}
} // namespace LossFunctions

namespace Maps {
const std::map<ActivationType, std::pair<ActivationFunction, ActivationFunction>> activation_map = {
    {ActivationType::SIGMOID, {ActivationFunctions::sigmoid, ActivationFunctions::sigmoid_derivative}},
    {ActivationType::RELU, {ActivationFunctions::relu, ActivationFunctions::relu_derivative}},
    {ActivationType::TANH, {ActivationFunctions::tanh_activation, ActivationFunctions::tanh_derivative}},
    {ActivationType::LINEAR, {ActivationFunctions::linear, ActivationFunctions::linear_derivative}},
    {ActivationType::SOFTMAX, {ActivationFunctions::softmax, ActivationFunctions::softmax_derivative}}};

const std::map<LossType, std::pair<LossFunction, LossDerivative>> loss_map = {
    {LossType::MSE, {LossFunctions::mse, LossFunctions::mse_derivative}},
    {LossType::BCE, {LossFunctions::bce, LossFunctions::bce_derivative}},
    {LossType::CCE, {LossFunctions::cce, LossFunctions::cce_derivative}}};
} // namespace Maps