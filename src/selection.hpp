#pragma once

#include "metrics.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// Holds the score of a model
// For classification the main metric is the error rate (1 - accuracy), and the tie-breaking metric is the Brier score
// For regression, both metrics are the error.
struct SelectionScore {
    Scalar main_metric = INF;
    Scalar tie_metric = INF;

    // Check if the score is valid (not NaN or Inf)
    bool valid() const { return std::isfinite(main_metric) && std::isfinite(tie_metric); }

    // Compares two SelectionScores, lower is better
    bool operator<(const SelectionScore& other) const {
        if (main_metric != other.main_metric) {
            return main_metric < other.main_metric;
        }
        return tie_metric < other.tie_metric;
    }

    // Returns the score of a run, which is the minimum over windows of the maximum error within the window
    // This score chooses the least worst model over windows of epochs, keeping the most stable one rather than the lucky one
    static SelectionScore from_run(const RunCurves& run) {
        const std::vector<Metrics>& epochs = run.holdout.epochs;
        const SelectionScore invalid{INF, INF};

        // Avoid scoring runs that are empty or invalid
        if (epochs.empty() || run.holdout.invalid) {
            return invalid;
        }
        if (std::ranges::any_of(epochs, [](const Metrics& e) { return !std::isfinite(e.error); })) {
            return invalid;
        }

        // Define a lambda to compute the score of a single epoch based on the run's task type
        auto epoch_score = [&run](const Metrics& e) {
            return run.track_accuracy() ? SelectionScore{Scalar(1) - e.accuracy, e.brier} : SelectionScore{e.error, e.error};
        };

        const size_t window_size = std::min(static_cast<size_t>(SELECTION_WINDOW), epochs.size());
        SelectionScore best_window_score = invalid;
        for (size_t i = 0; i + window_size <= epochs.size(); ++i) {
            // Worst score (highest error) inside this window
            SelectionScore window_worst = epoch_score(epochs[i]);
            for (size_t j = i + 1; j < i + window_size; ++j) {
                const SelectionScore current = epoch_score(epochs[j]);
                if (window_worst < current) {
                    window_worst = current;
                }
            }
            // Keep the window with the best worst score
            if (window_worst < best_window_score) {
                best_window_score = window_worst;
            }
        }
        return best_window_score;
    }

    // Average score of a set of SelectionScores
    static SelectionScore average_scores(const std::vector<SelectionScore>& scores) {
        // If any run is invalid, the score is invalid
        if (scores.empty() || std::ranges::any_of(scores, [](const SelectionScore& s) { return !s.valid(); })) {
            return {INF, INF};
        }
        const Scalar n = static_cast<Scalar>(scores.size());
        SelectionScore mean{0, 0};
        for (const SelectionScore& s : scores) {
            mean.main_metric += s.main_metric / n;
            mean.tie_metric += s.tie_metric / n;
        }
        return mean;
    }
};