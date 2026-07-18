#pragma once

#include "types.hpp"

// ACTIVATION FUNCTIONS

inline Matrix sigmoid(const Matrix &X) { return 1.0 / (1.0 + (-X.array()).exp()); }

inline Matrix sigmoid_derivative(const Matrix &X) {
    Matrix s = sigmoid(X);
    return s.array() * (1.0 - s.array());
}

inline Matrix relu(const Matrix &X) { return X.array().max(0.0); }

inline Matrix relu_derivative(const Matrix &X) { return (X.array() > 0.0).cast<Scalar>(); }

inline Matrix tanh_activation(const Matrix &X) { return X.array().tanh(); }

inline Matrix tanh_derivative(const Matrix &X) {
    Matrix t = tanh_activation(X);
    return 1.0 - (t.array() * t.array());
}

inline Matrix linear(const Matrix &X) { return X; }

inline Matrix linear_derivative(const Matrix &X) { return Matrix::Ones(X.rows(), X.cols()); }

inline Matrix softmax(const Matrix &X) {
    Matrix max_vals = X.colwise().maxCoeff();
    Matrix inputs = X.array() - max_vals.replicate(X.rows(), 1).array();

    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array() / sum.replicate(X.rows(), 1).array();
}

inline Matrix softmax_derivative(const Matrix &X) { return Matrix::Ones(X.rows(), X.cols()); }

// LOSS FUNCTIONS

inline double mse(const Matrix &target, const Matrix &prediction) { return (prediction - target).array().square().colwise().sum().mean(); }

inline Matrix mse_derivative(const Matrix &target, const Matrix &prediction) { return (prediction - target); }

inline double cce(const Matrix &target, const Matrix &prediction) {
    double epsilon = 1e-8;
    Matrix pred_clipped = prediction.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix &target, const Matrix &prediction) { return (prediction - target); }