#pragma once

#include "types.hpp"

#include <random>

inline std::mt19937& get_random_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

// ACTIVATION FUNCTIONS

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
    Matrix max_vals = X.colwise().maxCoeff();
    Matrix inputs = X.array() - max_vals.replicate(X.rows(), 1).array();

    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array() / sum.replicate(X.rows(), 1).array();
}

inline Matrix softmax_derivative(const Matrix& X) {
    return Matrix::Ones(X.rows(), X.cols());
}

// LOSS FUNCTIONS

inline double mse(const Matrix& target, const Matrix& prediction) {
    return (prediction - target).array().square().colwise().sum().mean();
}

inline Matrix mse_derivative(const Matrix& target, const Matrix& prediction) {
    return (prediction - target);
}

inline double cce(const Matrix& target, const Matrix& prediction) {
    double epsilon = 1e-8;
    Matrix pred_clipped = prediction.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
    return (prediction - target);
}

// ACCURACY FUNCTIONS

inline double classification_accuracy(const Matrix& target, const Matrix& prediction) {
    int correct_predictions = 0;
    int num_samples = target.cols();

    if (target.rows() == 1) {
        for (int i = 0; i < num_samples; ++i) {
            double pred_val = prediction(0, i) >= 0.5 ? 1.0 : 0.0;
            double target_val = target(0, i);

            if (pred_val == target_val) {
                correct_predictions++;
            }
        }
    } else {
        for (int i = 0; i < num_samples; ++i) {
            int target_class;
            int predicted_class;

            target.col(i).maxCoeff(&target_class);
            prediction.col(i).maxCoeff(&predicted_class);

            if (target_class == predicted_class) {
                correct_predictions++;
            }
        }
    }

    return static_cast<Scalar>(correct_predictions) / num_samples;
}