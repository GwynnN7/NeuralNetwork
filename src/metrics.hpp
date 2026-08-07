#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <print>
#include <vector>

namespace Metrics {
// Classification accuracy metric for binary and multi-class classification tasks
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

// Continuous and confidence-aware metric for classification tasks
inline Scalar brier_score(const Matrix& target, const Matrix& prediction) {
    if (target.cols() == 0) {
        return 0.0;
    }
    // Mean over samples of the squared distance between the predicted probability and the target
    return (prediction - target).array().square().colwise().sum().mean();
}
} // namespace Metrics

// Metrics for a single epoch
struct EpochMetric {
    Scalar error = 0;
    Scalar accuracy = 0;
    Scalar brier = 0;
    // Weight of this epoch for averaging across folds/batches
    Scalar weight = 0;
};

struct MetricsResult {
    std::vector<EpochMetric> epochs;
    // Set when training was interrupted by inf/NaN loss
    bool invalid = false;

    bool empty() const { return epochs.empty(); }
    size_t size() const { return epochs.size(); }
    Scalar last_error() const { return epochs.back().error; }

    void print(bool track_accuracy) const {
        if (empty()) {
            std::println(" • (no metrics recorded)");
            return;
        }
        auto mean = [this](Scalar EpochMetric::* field) {
            return std::accumulate(epochs.begin(), epochs.end(), Scalar(0), [field](Scalar acc, const EpochMetric& e) { return acc + e.*field; }) / epochs.size();
        };
        if (track_accuracy) {
            std::println(" • Accuracy:   {:.2f}%", mean(&EpochMetric::accuracy) * 100.0);
        }
        std::println(" • Error: {:.3f}", mean(&EpochMetric::error));
    }

    // Append metrics for a new epoch based on the provided target and prediction matrices, using the specified error function
    void append(const Matrix& target, const Matrix& prediction, const LossFunction& loss_func, bool track_accuracy = true) {
        epochs.push_back({loss_func(target, prediction),
                          track_accuracy ? Metrics::classification_accuracy(target, prediction) : Scalar(0),
                          track_accuracy ? Metrics::brier_score(target, prediction) : Scalar(0),
                          1});
    }

    // Accumulate one `batch` into the current epoch, weighted by the sample count
    void add(const Matrix& target, const Matrix& prediction, const LossFunction& loss_func, bool track_accuracy = true) {
        const Scalar n = static_cast<Scalar>(target.cols());
        const Scalar batch_error = loss_func(target, prediction) * n;
        const Scalar batch_accuracy = (track_accuracy ? Metrics::classification_accuracy(target, prediction) : Scalar(0)) * n;
        const Scalar batch_brier = (track_accuracy ? Metrics::brier_score(target, prediction) : Scalar(0)) * n;

        if (empty()) { // First batch of the epoch starts the accumulator
            epochs.push_back({batch_error, batch_accuracy, batch_brier, n});
            return;
        }

        epochs.back().error += batch_error;
        epochs.back().accuracy += batch_accuracy;
        epochs.back().brier += batch_brier;
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
            epochs[i].error += source.error;
            epochs[i].accuracy += source.accuracy;
            epochs[i].brier += source.brier;
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
            epoch.error /= epoch.weight;
            epoch.accuracy /= epoch.weight;
            epoch.brier /= epoch.weight;
        }
        epoch.weight = 1;
    }
};

// Model-selection score, lower is better. The error rate (1 - accuracy) is the task objective, and the Brier score is used for tie breaking. For regression the error is used
struct SelectionScore {
    Scalar error_rate = std::numeric_limits<Scalar>::infinity();
    Scalar brier = std::numeric_limits<Scalar>::infinity();

    bool operator<(const SelectionScore& other) const {
        if (error_rate != other.error_rate) {
            return error_rate < other.error_rate;
        }
        return brier < other.brier;
    }
    bool valid() const { return std::isfinite(error_rate) && std::isfinite(brier); }
};

struct SplitResults {
    MetricsResult train_metrics;
    MetricsResult test_metrics;
    TaskType task;

    SplitResults(TaskType task_type = TaskType::REGRESSION) : task(task_type) {};

    // Call the add method for both train_metrics and test_metrics
    void add(const SplitResults& other) {
        if (task != other.task) {
            throw std::invalid_argument("Cannot add SplitResults with different task types");
        }
        train_metrics.add(other.train_metrics);
        test_metrics.add(other.test_metrics);
    }

    // Call the average method for both train_metrics and test_metrics
    void average() {
        train_metrics.average();
        test_metrics.average();
    }

    void print() const {
        std::println("\nTraining Set Metrics:");
        train_metrics.print(track_accuracy());
        std::println("Test Set Metrics:");
        test_metrics.print(track_accuracy());
    }

    bool track_accuracy() const { return task == TaskType::CLASSIFICATION; }

    // The score is the lowest, over all sliding windows, of the worst penalty within the window
    // To avoid comparing different error functions that are on different scales, the error rate is used for model selection. Regression uses the error
    SelectionScore score() const {
        const auto& epochs = test_metrics.epochs;
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        const SelectionScore invalid{inf, inf};

        // If the test metrics are empty or invalid, the model is invalid
        if (epochs.empty() || test_metrics.invalid) {
            return invalid;
        }
        // If any epoch has NaN or Inf error, the model is invalid
        if (std::ranges::any_of(epochs, [](const EpochMetric& e) { return !std::isfinite(e.error); })) {
            return invalid;
        }

        auto epoch_score = [this](const EpochMetric& e) {
            return track_accuracy() ? SelectionScore{Scalar(1) - e.accuracy, e.brier} : SelectionScore{e.error, e.error};
        };

        // Determine the size of the sliding window, which is the minimum of SELECTION_WINDOW and the number of epochs
        const size_t window_size = std::min(static_cast<size_t>(SELECTION_WINDOW), epochs.size());
        SelectionScore best_window_score = invalid;
        for (size_t i = 0; i + window_size <= epochs.size(); ++i) {
            // Find the worst penalty (best score) in the current window
            SelectionScore window_best = epoch_score(epochs[i]);
            for (size_t j = i + 1; j < i + window_size; ++j) {
                const SelectionScore current = epoch_score(epochs[j]);
                if (window_best < current) {
                    window_best = current;
                }
            }
            // Update the best window score if the current window's worst is lower
            if (window_best < best_window_score) {
                best_window_score = window_best;
            }
        }
        return best_window_score;
    }

    // Compare this SplitResults with another to determine if this one is better than the other
    bool is_better_than(const SplitResults& other) const {
        const SelectionScore this_score = score();
        const SelectionScore other_score = other.score();
        if (!this_score.valid()) {
            return false; // Never prefer an invalid model
        }
        return !other_score.valid() || this_score < other_score;
    }
};
