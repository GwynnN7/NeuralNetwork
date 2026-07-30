#pragma once

#include "types.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <print>
#include <vector>

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
    Matrix inputs = X.rowwise() - X.colwise().maxCoeff(); // Subtract the max value in each column for numerical stability

    inputs = inputs.array().exp();
    Matrix sum = inputs.colwise().sum();
    return inputs.array() / sum.replicate(X.rows(), 1).array();
}

inline Matrix softmax_derivative(const Matrix& X) {
    return linear_derivative(X); // The derivative of softmax is handled in the CCE loss function
}

const std::map<ActivationType, std::pair<ActivationFunction, ActivationFunction>> activation_map = {
    {ActivationType::SIGMOID, {sigmoid, sigmoid_derivative}},
    {ActivationType::RELU, {relu, relu_derivative}},
    {ActivationType::TANH, {tanh_activation, tanh_derivative}},
    {ActivationType::LINEAR, {linear, linear_derivative}},
    {ActivationType::SOFTMAX, {softmax, softmax_derivative}}};

// Loss Functions

inline double mse(const Matrix& target, const Matrix& prediction) {
    return (prediction - target).array().square().colwise().sum().mean(); // Averaged across output neurons and samples
}

inline Matrix mse_derivative(const Matrix& target, const Matrix& prediction) {
    return 2 * (prediction - target);
}

inline double cce(const Matrix& target, const Matrix& prediction) {
    double epsilon = 1e-8; // Small value to prevent log(0)
    Matrix pred_clipped = prediction.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    return -(target.cwiseProduct(pred_clipped.array().log().matrix())).colwise().sum().mean();
}

inline Matrix cce_derivative(const Matrix& target, const Matrix& prediction) {
    return (prediction - target);
}

const std::map<LossType, std::pair<LossFunction, LossDerivative>> loss_map = {
    {LossType::MSE, {mse, mse_derivative}},
    {LossType::CCE, {cce, cce_derivative}}};

// Metrics Functions

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

struct MetricsResult {
    std::vector<Scalar> loss;
    std::vector<Scalar> accuracy;
    std::vector<int> occurrences;

    void print(bool is_regression) const {
        if (!is_regression) {
            std::println(" • Accuracy:   {:.2f}%", std::accumulate(accuracy.begin(), accuracy.end(), 0.0) / accuracy.size() * 100.0);
        }
        std::println(" • Loss: {:.3f}", std::accumulate(loss.begin(), loss.end(), 0.0) / loss.size());
    }

    void add(const Matrix& target, const Matrix& prediction, LossFunction loss_func) {
        loss.push_back(loss_func(target, prediction));
        accuracy.push_back(classification_accuracy(target, prediction));
        occurrences.push_back(1);
    }

    void add(const MetricsResult& other) {
        loss.insert(loss.end(), other.loss.begin(), other.loss.end());
        accuracy.insert(accuracy.end(), other.accuracy.begin(), other.accuracy.end());
        occurrences.insert(occurrences.end(), other.occurrences.begin(), other.occurrences.end());
    }

    void merge(const MetricsResult& other) {
        for (size_t i = 0; i < other.loss.size(); ++i) {
            if (i < loss.size()) {
                loss[i] += other.loss[i];
                accuracy[i] += other.accuracy[i];
                occurrences[i] += other.occurrences[i];
            } else {
                loss.push_back(other.loss[i]);
                accuracy.push_back(other.accuracy[i]);
                occurrences.push_back(other.occurrences[i]);
            }
        }
    }

    void merge(const Matrix& target, const Matrix& prediction, LossFunction loss_func) {
        Scalar batch_loss = loss_func(target, prediction);
        Scalar batch_accuracy = classification_accuracy(target, prediction);

        if (loss.empty()) {
            add(target, prediction, loss_func);
            return;
        }

        loss.back() += batch_loss;
        accuracy.back() += batch_accuracy;
        occurrences.back() += 1;
    }

    void average() {
        for (size_t i = 0; i < loss.size(); ++i) {
            loss[i] /= occurrences[i];
            accuracy[i] /= occurrences[i];
            occurrences[i] = 1;
        }
    }
};

struct SplitResults {
    MetricsResult train_metrics;
    MetricsResult test_metrics;

    void merge(const SplitResults& other) {
        train_metrics.merge(other.train_metrics);
        test_metrics.merge(other.test_metrics);
    }

    void average() {
        train_metrics.average();
        test_metrics.average();
    }

    bool operator>(const SplitResults& other) const {
        if (other.test_metrics.loss.empty()) {
            return true;
        }

        auto this_min = std::min_element(test_metrics.loss.begin(), test_metrics.loss.end());
        auto other_min = std::min_element(other.test_metrics.loss.begin(), other.test_metrics.loss.end());

        return *this_min <= *other_min;
    }

    int get_best_index() const {
        auto min_it = std::min_element(test_metrics.loss.begin(), test_metrics.loss.end());
        return std::distance(test_metrics.loss.begin(), min_it);
    }
};