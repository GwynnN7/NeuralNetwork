#pragma once

#include "types.hpp"

// Activation Functions

namespace ActivationFunctions {
inline Matrix sigmoid(const Matrix& X) {
    return 1.0 / (1.0 + (-X.array()).exp());
}

inline Matrix sigmoid_derivative(const Matrix& X) {
    Matrix s = sigmoid(X);
    return s.array() * (1.0 - s.array());
}

inline Matrix relu(const Matrix& X) {
    return X.array().max(0.0);
}

inline Matrix relu_derivative(const Matrix& X) {
    return (X.array() > 0.0).cast<Scalar>();
}

inline Matrix tanh_activation(const Matrix& X) {
    return X.array().tanh();
}

inline Matrix tanh_derivative(const Matrix& X) {
    Matrix t = tanh_activation(X);
    return 1.0 - (t.array() * t.array());
}

inline Matrix linear(const Matrix& X) {
    return X;
}

inline Matrix linear_derivative(const Matrix& X) {
    return Matrix::Ones(X.rows(), X.cols());
}

inline Matrix softmax(const Matrix& X) {
    Matrix inputs = X.rowwise() - X.colwise().maxCoeff(); // Subtract the max value in each column for numerical stability

    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array() / sum.replicate(X.rows(), 1).array();
}

inline Matrix softmax_derivative(const Matrix& X) {
    return linear_derivative(X); // The derivative of softmax is handled in the CCE loss function
}
} // namespace ActivationFunctions

namespace LossFunctions {
inline Scalar mse(const Matrix& target, const Matrix& prediction) {
    return (prediction - target).array().square().colwise().sum().mean(); // Averaged across output neurons and samples
}

inline Matrix mse_derivative(const Matrix& target, const Matrix& prediction) {
    return 2 * (prediction - target);
}

inline Scalar cce(const Matrix& target, const Matrix& prediction) {
    Scalar epsilon = 1e-8; // Small value to prevent log(0)
    Matrix pred_clipped = prediction.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
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
    {LossType::CCE, {LossFunctions::cce, LossFunctions::cce_derivative}}};
} // namespace Maps