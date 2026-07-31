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
    std::vector<int> occurrences;

    void print(bool is_regression) const {
        if (!is_regression) {
            std::println(" • Accuracy:   {:.2f}%", std::accumulate(accuracy.begin(), accuracy.end(), 0.0) / accuracy.size() * 100.0);
        }
        std::println(" • Loss: {:.3f}", std::accumulate(loss.begin(), loss.end(), 0.0) / loss.size());
    }

    void add(const Matrix& target, const Matrix& prediction, LossFunction loss_func) {
        loss.push_back(loss_func(target, prediction));
        accuracy.push_back(Metrics::classification_accuracy(target, prediction));
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
        Scalar batch_accuracy = Metrics::classification_accuracy(target, prediction);

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