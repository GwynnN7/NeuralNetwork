#pragma once

#include "functions.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <vector>

// Measures computed from a set of targets and predictions
namespace Measure {
// Accuracy for binary and multi-class tasks
QUERY inline Scalar accuracy(const Matrix& target, const Matrix& prediction) {
    int correct_predictions = 0;
    const int num_samples = static_cast<int>(target.cols());
    if (num_samples == 0) {
        return 0.0;
    }

    if (target.rows() == 1) { // Binary classification
        for (int i = 0; i < num_samples; ++i) {
            const int pred_positive = get_task_class(prediction, i); // Get the predicted class
            const int target_positive = get_task_class(target, i);   // Get the target class

            // Count correct predictions
            if (pred_positive == target_positive) {
                correct_predictions++;
            }
        }
    } else { // Multi-class classification
        for (int i = 0; i < num_samples; ++i) {
            const int target_class = get_task_class(target, i);        // Get the target class
            const int predicted_class = get_task_class(prediction, i); // Get the predicted class

            // Count correct predictions
            if (target_class == predicted_class) {
                correct_predictions++;
            }
        }
    }

    return static_cast<Scalar>(correct_predictions) / num_samples;
}

// Measure for classification tasks with confidence
QUERY inline Scalar brier(const Matrix& target, const Matrix& prediction) {
    if (target.cols() == 0) {
        return 0.0;
    }
    // Mean over samples of the squared distance between the predicted probability and the target
    return (prediction - target).array().square().colwise().sum().mean();
}
} // namespace Measure

// Every metric measured at each step of training
struct Metrics {
    Scalar mse = 0;
    Scalar error = 0;
    Scalar accuracy = 0;
    Scalar brier = 0;
    // Number of samples that contributed to this metric
    Scalar weight = 0;

    // Compute the metrics for a given set of targets and predictions
    QUERY static Metrics evaluate(const Matrix& target, const Matrix& prediction, LossType loss_type, bool track_accuracy, bool weighted = false) {
        const Scalar n = weighted ? static_cast<Scalar>(target.cols()) : Scalar(1);
        const Scalar error = Lookup::loss_for(loss_type)->function(target, prediction);
        return Metrics{
            .mse = (loss_type == LossType::MSE ? error : LossFunctions::mse(target, prediction)) * n,
            .error = error * n,
            .accuracy = (track_accuracy ? Measure::accuracy(target, prediction) : Scalar(0)) * n,
            .brier = (track_accuracy ? Measure::brier(target, prediction) : Scalar(0)) * n,
            .weight = weighted ? n : Scalar(1)};
    }

    void operator+=(const Metrics& other) noexcept {
        mse += other.mse;
        error += other.error;
        accuracy += other.accuracy;
        brier += other.brier;
        weight += other.weight;
    }

    // Average the metrics over the number of samples that contributed
    void normalize() noexcept {
        if (weight != 0) {
            mse /= weight;
            error /= weight;
            accuracy /= weight;
            brier /= weight;
        }
        weight = 1;
    }
};

// The metrics of one training run
struct LearningCurve {
    std::vector<Metrics> epochs;
    // Set when training was interrupted by an inf/NaN error
    bool invalid = false;

    QUERY bool empty() const noexcept { return epochs.empty(); }
    QUERY size_t size() const noexcept { return epochs.size(); }
    QUERY Scalar last_error() const noexcept { return epochs.back().error; }

    // Metrics of a single epoch, by default the last one
    QUERY const Metrics& at(int epoch = -1) const noexcept {
        return (epoch >= 0 && static_cast<size_t>(epoch) < size()) ? epochs[epoch] : epochs.back();
    }

    // Record an epoch
    void append_epoch(const Matrix& target, const Matrix& prediction, LossType loss_type, bool track_accuracy = true) {
        epochs.push_back(Metrics::evaluate(target, prediction, loss_type, track_accuracy));
    }

    // Append epochs from another LearningCurve
    void append_epochs(const LearningCurve& other) {
        epochs.insert(epochs.end(), other.epochs.begin(), other.epochs.end());
        invalid = invalid || other.invalid;
    }

    // Accumulate one batch into the current epoch
    void add_batch(const Matrix& target, const Matrix& prediction, LossType loss_type, bool track_accuracy = true) {
        const Metrics batch = Metrics::evaluate(target, prediction, loss_type, track_accuracy, true);
        if (empty()) { // First batch of the epoch starts the accumulator
            epochs.push_back(batch);
        } else {
            epochs.back() += batch;
        }
    }

    // Turn the epoch in progress from a sum over its batches into an average
    void normalize_epoch() {
        if (!empty()) {
            epochs.back().normalize();
        }
    }
};

// The result of a single training run
struct RunCurves {
    LearningCurve training;
    LearningCurve holdout;
    TaskType task;
    // The epoch that was selected as the best by early stopping
    int best_epoch = 0;

    explicit RunCurves(TaskType task_type = TaskType::REGRESSION) : task(task_type) {};

    QUERY bool track_accuracy() const noexcept { return task == TaskType::CLASSIFICATION; }
};
