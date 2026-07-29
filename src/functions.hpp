#pragma once

#include "types.hpp"

#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Random Functions

inline std::mt19937& get_random_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

inline void set_random_seed(unsigned int seed) {
    get_random_generator().seed(seed);
}

// Activation Functions

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

const std::map<ActivationType, std::pair<std::function<Matrix(const Matrix&)>, std::function<Matrix(const Matrix&)>>> activation_map = {
    {ActivationType::SIGMOID, {sigmoid, sigmoid_derivative}},
    {ActivationType::RELU, {relu, relu_derivative}},
    {ActivationType::TANH, {tanh_activation, tanh_derivative}},
    {ActivationType::LINEAR, {linear, linear_derivative}},
    {ActivationType::SOFTMAX, {softmax, softmax_derivative}}};

// Loss Functions

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

const std::map<LossType, std::pair<std::function<double(const Matrix&, const Matrix&)>, std::function<Matrix(const Matrix&, const Matrix&)>>> loss_map = {
    {LossType::MSE, {mse, mse_derivative}},
    {LossType::CCE, {cce, cce_derivative}}};

// Accuracy Functions

inline double classification_accuracy(const Matrix& target, const Matrix& prediction) {
    int correct_predictions = 0;
    int num_samples = target.cols();

    if (target.rows() == 1) { // Binary classification
        for (int i = 0; i < num_samples; ++i) {
            double pred_val = prediction(0, i) >= 0.5 ? 1.0 : 0.0; // Convert prediction to binary class
            double target_val = target(0, i);                      // Get the target class

            // Count correct predictions
            if (pred_val == target_val) {
                correct_predictions++;
            }
        }
    } else { // Multi-class classification
        for (int i = 0; i < num_samples; ++i) {
            int target_class, predicted_class;

            target.col(i).maxCoeff(&target_class);        // Get the index of the maximum value in the target column (target class)
            prediction.col(i).maxCoeff(&predicted_class); // Get the index of the maximum value in the prediction column (the predicted class)

            // Count correct predictions
            if (target_class == predicted_class) {
                correct_predictions++;
            }
        }
    }

    return static_cast<Scalar>(correct_predictions) / num_samples;
}

// Calculate the classification confidence as a percentage with mean and variance
inline std::string classification_confidence(const Matrix& target, const Matrix& prediction) {
    int num_samples = target.cols();
    std::vector<Scalar> confidences;
    confidences.reserve(num_samples); // Reserve space for efficiency

    if (target.rows() == 1) { // Binary classification
        for (int i = 0; i < num_samples; ++i) {
            Scalar target_val = target(0, i);                                       // Get the target class
            Scalar pred_val = prediction(0, i);                                     // Get the predicted probability for the positive class
            confidences.push_back(target_val == 1.0 ? pred_val : (1.0 - pred_val)); // Store the predicted probability for the target class
        }
    } else { // Multi-class classification
        for (int i = 0; i < num_samples; ++i) {
            int target_class, predicted_class;
            target.col(i).maxCoeff(&target_class);              // Get the index of the maximum value in the target column (target class)
            prediction.col(i).maxCoeff(&predicted_class);       // Get the index of the maximum value in the prediction column (predicted class)
            confidences.push_back(prediction(target_class, i)); // Store the predicted probability for the target class
        }
    }

    // Calculate mean and variance of the confidences
    Scalar mean_confidence = std::accumulate(confidences.begin(), confidences.end(), 0.0) / num_samples;
    Scalar variance = std::accumulate(confidences.begin(), confidences.end(), 0.0, [&](Scalar sum, Scalar val) {
                          return sum + (val - mean_confidence) * (val - mean_confidence);
                      }) /
                      num_samples;

    return to_string_rounded(mean_confidence * 100.0, 2) + "% ± " + to_string_rounded(variance * 100.0, 2) + "%";
}