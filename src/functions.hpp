#pragma once

#include "types.hpp"

// Activation Functions

namespace ActivationFunctions {
inline Matrix sigmoid(const Matrix& X) {
    // f(x) = 1 / (1 + e^(-x))
    return 1.0 / (1.0 + (-X.array()).exp());
}

inline Matrix sigmoid_derivative(const Matrix& X) {
    // f'(x) = f(x) * (1 - f(x))
    Matrix s = sigmoid(X);
    return s.array() * (1.0 - s.array());
}

inline Matrix relu(const Matrix& X) {
    // f(x) = max(0, x)
    return X.array().max(0.0);
}

inline Matrix relu_derivative(const Matrix& X) {
    // f'(x) = 1 if x > 0, else 0
    return (X.array() > 0.0).cast<Scalar>();
}

inline Matrix tanh_activation(const Matrix& X) {
    // f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = tanh(x)
    return X.array().tanh();
}

inline Matrix tanh_derivative(const Matrix& X) {
    // f'(x) = 1 - f(x)^2
    Matrix t = tanh_activation(X);
    return 1.0 - (t.array() * t.array());
}

inline Matrix linear(const Matrix& X) {
    // f(x) = x
    return X;
}

inline Matrix linear_derivative(const Matrix& X) {
    // f'(x) = 1
    return Matrix::Ones(X.rows(), X.cols());
}

inline Matrix softmax(const Matrix& X) {
    // f(x) = e^(x_i) / sum(e^(x_j))
    Matrix inputs = X.rowwise() - X.colwise().maxCoeff(); // Avoid e^x overflow
    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array() / sum.replicate(X.rows(), 1).array();
}

inline Matrix softmax_derivative(const Matrix& X) {
    // Passthrough, the calculation is simplified by CCE.
    return linear_derivative(X);
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
    Matrix pred_clipped = prediction.cwiseMax(EPSILON).cwiseMin(1.0 - EPSILON);
    // f = -(1 / N) * sum(y * log(y*) + (1 - y) * log(1 - y*))
    return -(target.array() * pred_clipped.array().log() +
             (1.0 - target.array()) * (1.0 - pred_clipped.array()).log())
                .sum() /
           target.cols();
}

inline Matrix bce_derivative(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(EPSILON).cwiseMin(1.0 - EPSILON);
    // f' = (y* - y) / (y* * (1 - y*))
    return (pred_clipped - target).array() / (pred_clipped.array() * (1.0 - pred_clipped.array()));
}

inline Scalar cce(const Matrix& target, const Matrix& prediction) {
    Matrix pred_clipped = prediction.cwiseMax(EPSILON).cwiseMin(1.0 - EPSILON);
    // f = -(1 / N) * sum(y * log(y*))
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
    // Calculates the combined derivative of CCE + Softmax: f' = y* - y
    return (prediction - target);
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