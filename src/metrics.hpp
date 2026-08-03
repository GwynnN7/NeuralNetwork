#pragma once

#include "types.hpp"

#include <algorithm>
#include <numeric>
#include <print>
#include <vector>

namespace Metrics {
inline Scalar classification_accuracy(const Matrix& target, const Matrix& prediction) {
    int correct_predictions = 0;
    int num_samples = target.cols();

    if (target.rows() == 1) { // Binary classification
        for (int i = 0; i < num_samples; ++i) {
            Scalar pred_val = prediction(0, i) >= 0.5 ? 1.0 : 0.0; // Convert prediction to binary class
            Scalar target_val = target(0, i);                      // Get the target class

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
} // namespace Metrics

struct MetricsResult {
    std::vector<Scalar> loss;
    std::vector<Scalar> accuracy;
    std::vector<int> occurrences; // Number of occurrences for each epoch (used for averaging)

    void print(TaskType task) const {
        if (task == TaskType::CLASSIFICATION) {
            std::println(" • Accuracy:   {:.2f}%", std::accumulate(accuracy.begin(), accuracy.end(), 0.0) / accuracy.size() * 100.0);
        }
        std::println(" • Loss: {:.3f}", std::accumulate(loss.begin(), loss.end(), 0.0) / loss.size());
    }

    // Append metrics for a new epoch based on the provided target and prediction matrices, using the specified loss function
    void append(const Matrix& target, const Matrix& prediction, LossFunction loss_func) {
        loss.push_back(loss_func(target, prediction));
        accuracy.push_back(Metrics::classification_accuracy(target, prediction));
        occurrences.push_back(1);
    }

    // Append metrics from another MetricsResult instance
    void append(const MetricsResult& other) {
        loss.insert(loss.end(), other.loss.begin(), other.loss.end());
        accuracy.insert(accuracy.end(), other.accuracy.begin(), other.accuracy.end());
        occurrences.insert(occurrences.end(), other.occurrences.begin(), other.occurrences.end());
    }

    // Add metrics from another MetricsResult instance, accumulating values for the same epochs
    void add(const MetricsResult& other) {
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

    // Add metrics for a new epoch based on the provided target and prediction matrices, using the specified loss function, accumulating values for the same epochs
    void add(const Matrix& target, const Matrix& prediction, LossFunction loss_func) {
        Scalar batch_loss = loss_func(target, prediction);
        Scalar batch_accuracy = Metrics::classification_accuracy(target, prediction);

        // If this is the first epoch, initialize the metrics
        if (loss.empty()) {
            append(target, prediction, loss_func);
            return;
        }

        loss.back() += batch_loss;
        accuracy.back() += batch_accuracy;
        occurrences.back() += 1;
    }

    // Average the accumulated metrics for each epoch using the number of occurrences
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

    // Call the append method for both train_metrics and test_metrics
    void append(const Matrix& train_target, const Matrix& train_prediction, const Matrix& test_target, const Matrix& test_prediction, LossFunction loss_func) {
        train_metrics.append(train_target, train_prediction, loss_func);
        test_metrics.append(test_target, test_prediction, loss_func);
    }

    // Call the add method for both train_metrics and test_metrics
    void add(const SplitResults& other) {
        train_metrics.add(other.train_metrics);
        test_metrics.add(other.test_metrics);
    }

    // Call the average method for both train_metrics and test_metrics
    void average() {
        train_metrics.average();
        test_metrics.average();
    }

    void print(TaskType task) const {
        std::println("\nTraining Set Metrics:");
        train_metrics.print(task);
        std::println("Test Set Metrics:");
        test_metrics.print(task);
    }

    // Overload the '>' operator to compare SplitResults based on the best test metric
    bool operator>(const SplitResults& other) const {
        if (other.get_metric().empty())
            return true;
        if (get_metric().empty())
            return false;

        return get_best_metric() < other.get_best_metric();
    }

  private:
    // Return the test metrics (loss) for comparison
    const std::vector<Scalar>& get_metric() const {
        return test_metrics.loss;
    }

    // Return the best test metric (lowest loss) for comparison
    Scalar get_best_metric() const {
        const auto& metric = get_metric();
        return *std::min_element(metric.begin(), metric.end());
    }
};