#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <print>
#include <vector>

namespace Metrics {
inline Scalar classification_accuracy(const Matrix& target, const Matrix& prediction) {
    int correct_predictions = 0;
    int num_samples = static_cast<int>(target.cols());
    if (num_samples == 0) {
        return 0.0;
    }

    if (target.rows() == 1) { // Binary classification
        for (int i = 0; i < num_samples; ++i) {
            bool pred_positive = prediction(0, i) >= Scalar(0.5); // Convert prediction to binary class
            bool target_positive = target(0, i) >= Scalar(0.5);   // Get the target class

            // Count correct predictions
            if (pred_positive == target_positive) {
                correct_predictions++;
            }
        }
    } else { // Multi-class classification
        for (int i = 0; i < num_samples; ++i) {
            Eigen::Index target_class, predicted_class;

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

// Metrics for a single epoch
struct EpochMetric {
    Scalar loss = 0;
    Scalar accuracy = 0;
    // Weight of this epoch for averaging across folds/batches
    Scalar weight = 0;
};

struct MetricsResult {
    std::vector<EpochMetric> epochs;
    // Set when training was interrupted by inf/NaN loss
    bool invalid = false;

    bool empty() const { return epochs.empty(); }
    size_t size() const { return epochs.size(); }
    Scalar last_loss() const { return epochs.back().loss; }

    void print(TaskType task) const {
        if (empty()) {
            std::println(" • (no metrics recorded)");
            return;
        }
        auto mean = [this](Scalar EpochMetric::* field) {
            return std::accumulate(epochs.begin(), epochs.end(), Scalar(0), [field](Scalar acc, const EpochMetric& e) { return acc + e.*field; }) / epochs.size();
        };
        if (task == TaskType::CLASSIFICATION) {
            std::println(" • Accuracy:   {:.2f}%", mean(&EpochMetric::accuracy) * 100.0);
        }
        std::println(" • Loss: {:.3f}", mean(&EpochMetric::loss));
    }

    // Append metrics for a new epoch based on the provided target and prediction matrices, using the specified loss function
    void append(const Matrix& target, const Matrix& prediction, const LossFunction& loss_func, bool track_accuracy = true) {
        epochs.push_back({loss_func(target, prediction), track_accuracy ? Metrics::classification_accuracy(target, prediction) : Scalar(0), 1});
    }

    // Accumulate one `batch` into the current epoch, weighted by the sample count
    void add(const Matrix& target, const Matrix& prediction, const LossFunction& loss_func, bool track_accuracy = true) {
        const Scalar n = static_cast<Scalar>(target.cols());
        const Scalar batch_loss = loss_func(target, prediction) * n;
        const Scalar batch_accuracy = (track_accuracy ? Metrics::classification_accuracy(target, prediction) : Scalar(0)) * n;

        if (empty()) { // First batch of the epoch starts the accumulator
            epochs.push_back({batch_loss, batch_accuracy, n});
            return;
        }

        epochs.back().loss += batch_loss;
        epochs.back().accuracy += batch_accuracy;
        epochs.back().weight += n;
    }

    // Append metrics from another MetricsResult instance
    void append(const MetricsResult& other) {
        epochs.insert(epochs.end(), other.epochs.begin(), other.epochs.end());
        invalid = invalid || other.invalid;
    }

    // Combine another fold data into this accumulator, aligning by epoch.
    void add(const MetricsResult& other) {
        if (other.empty()) {
            return;
        }
        if (empty()) {
            *this = other; // First fold simply initializes the accumulator
            return;
        }

        invalid = invalid || other.invalid;

        if (invalid) {
            // If either fold is invalid, keep the invalid state and truncate to the shorter length
            epochs.resize(std::min(size(), other.size()));
        } else {
            // Pad with last value if this fold stopped earlier than 'other'
            const EpochMetric last = epochs.back();
            epochs.resize(std::max(size(), other.size()), last);
        }

        for (size_t i = 0; i < size(); ++i) {
            // Use the last value if 'other' fold stopped earlier than this, otherwise use the corresponding value
            const EpochMetric& source = (i < other.size()) ? other.epochs[i] : other.epochs.back();
            epochs[i].loss += source.loss;
            epochs[i].accuracy += source.accuracy;
            epochs[i].weight += 1;
        }
    }

    // Average the accumulated metrics for each epoch using the accumulated weights
    void average() {
        for (EpochMetric& epoch : epochs) {
            average(epoch);
        }
    }

    // Average only the most recent epoch
    void average_last() {
        if (!empty()) {
            average(epochs.back());
        }
    }

  private:
    // Average the metrics of a single epoch using its accumulated weight
    static void average(EpochMetric& epoch) {
        if (epoch.weight != 0) {
            epoch.loss /= epoch.weight;
            epoch.accuracy /= epoch.weight;
        }
        epoch.weight = 1;
    }
};

struct SplitResults {
    MetricsResult train_metrics;
    MetricsResult test_metrics;

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

    // The score is the lowest, over all sliding windows, of the highest loss within the window.
    Scalar score() const {
        const auto& epochs = test_metrics.epochs;
        // +infinity is used to indicate that the model is `invalid`
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // If the test metrics are empty or invalid, return `invalid` score
        if (epochs.empty() || test_metrics.invalid) {
            return inf;
        }
        // If any epoch has NaN or Inf loss, return `invalid` score
        if (std::ranges::any_of(epochs, [](const EpochMetric& e) { return !std::isfinite(e.loss); })) {
            return inf;
        }

        // Determine the size of the sliding window, which is the minimum of SELECTION_WINDOW and the number of epochs
        const size_t window_size = std::min(static_cast<size_t>(SELECTION_WINDOW), epochs.size());
        Scalar best_window_score = inf;
        for (size_t i = 0; i + window_size <= epochs.size(); ++i) {
            // Find the maximum loss in the current window
            const auto window_max = std::ranges::max_element(epochs.begin() + i, epochs.begin() + i + window_size, {}, &EpochMetric::loss);
            // Update the best window score if the current window's maximum is lower
            best_window_score = std::min(best_window_score, window_max->loss);
        }
        return best_window_score;
    }

    // Compare this SplitResults with another to determine if this one is better than the other
    bool is_better_than(const SplitResults& other) const {
        const Scalar this_score = score();
        const Scalar other_score = other.score();
        if (!std::isfinite(this_score)) {
            return false; // Never prefer an invalid model
        }
        return !std::isfinite(other_score) || this_score < other_score;
    }
};
