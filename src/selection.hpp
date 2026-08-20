#pragma once

#include "metrics.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>

/*
Scores used for model selection
For classification the main metric is the error rate (1 - accuracy), and the tie-breaking metric is the MSE
For regression the main metric is the MEE, and the tie-breaking metric is the MSE

The total score of a run is the minimum "custom" Upper Confidence Bound (called UCB from now on) across all windows of epochs
The UCB of a window is the average of the metric summed with half the standard deviation of the metric
This method avoids selecting models that have a noisy validation curve with a possible low point, thus preferring more stable and smooth models
*/

struct SelectionScore {
    Stats main_metric = Stats::inf();
    Stats tie_metric = Stats::inf();

    // Check if the score is valid (not NaN or Inf)
    OUT bool valid() const noexcept { return std::isfinite(main_metric.mean) && std::isfinite(tie_metric.mean); }

    // Compares two SelectionScores, lower is better
    OUT bool operator<(const SelectionScore& other) const noexcept {
        // Valid scores are better than invalid scores
        if (!valid() && other.valid()) {
            return false;
        }
        if (valid() && !other.valid()) {
            return true;
        }

        if (SelectionScore::ucb(main_metric) != SelectionScore::ucb(other.main_metric)) {
            return SelectionScore::ucb(main_metric) < SelectionScore::ucb(other.main_metric);
        }
        return SelectionScore::ucb(tie_metric) < SelectionScore::ucb(other.tie_metric);
    }

    // Returns the total score of a run
    OUT static SelectionScore from_run(const RunCurves& run, int selection_window) {
        const std::vector<Metrics>& epochs = run.holdout.epochs;
        const SelectionScore invalid{Stats::inf(), Stats::inf()};

        // Avoid scoring runs that are empty or invalid
        if (run.holdout.invalid) {
            return invalid;
        }
        if (epochs.empty() || std::ranges::any_of(epochs, [](const Metrics& e) { return !std::isfinite(e.error); })) {
            return invalid;
        }

        // Lambda to retrieve the score of a single epoch based on the run's task type
        auto epoch_score = [&run](const Metrics& e) {
            return run.track_accuracy() ? std::pair<Scalar, Scalar>(Scalar(1) - e.accuracy, e.mse) : std::pair<Scalar, Scalar>(e.mee, e.mse);
        };

        if (selection_window <= 0) {
            return invalid;
        }
        const size_t window_size = std::min(static_cast<size_t>(selection_window), epochs.size());
        SelectionScore best_score = {Stats::inf(), Stats::inf()};
        // Compute the score for each window of epochs and keep track of the best score
        for (size_t i = 0; i + window_size <= epochs.size(); ++i) {
            std::vector<std::pair<Scalar, Scalar>> window_stats;
            for (size_t j = i; j < i + window_size; ++j) {
                window_stats.push_back(epoch_score(epochs[j]));
            }
            // Compute the mean and std of the main and tie metrics for the current window
            SelectionScore window_score{
                Stats::of(window_stats | std::views::transform([](const auto& p) { return p.first; })),
                Stats::of(window_stats | std::views::transform([](const auto& p) { return p.second; }))};
            best_score = std::min(best_score, window_score);
        }
        return best_score;
    }

    // Average score of a set of SelectionScores
    OUT static SelectionScore average_scores(const std::vector<SelectionScore>& scores) {
        // If any run is invalid, the score is invalid
        if (scores.empty() || std::ranges::any_of(scores, [](const SelectionScore& s) { return !s.valid(); })) {
            return {Stats::inf(), Stats::inf()};
        }
        // Lambda to compute the average of a set of Stats, for both the mean and std
        auto average_stats = [](const auto& stats) {
            std::size_t total_samples = 0;
            Scalar combined_mean = 0.0;

            // Weighted combined mean
            for (const auto& s : stats) {
                if (s.samples <= 0) {
                    continue;
                }

                total_samples += s.samples;
                combined_mean += static_cast<Scalar>(s.samples) * s.mean;
            }

            if (total_samples == 0) {
                return Stats{};
            }

            combined_mean /= static_cast<Scalar>(total_samples);

            // Can't just average the stds, need to compute the combined std
            // Law of total variance: SST = SSW + SSB
            Scalar total_ss = 0.0;
            for (const auto& s : stats) {
                if (s.samples <= 0) {
                    continue;
                }

                // Sum of Squares within each group: (n - 1) * variance
                const Scalar within_ss = static_cast<Scalar>(s.samples - 1) * s.std * s.std;

                // Sum of Squares between groups: n * (mean - combined_mean)^2
                const Scalar diff = s.mean - combined_mean;
                const Scalar between_ss = static_cast<Scalar>(s.samples) * diff * diff;

                total_ss += within_ss + between_ss;
            }

            // Combined std
            const Scalar combined_std = total_samples > 1 ? std::sqrt(total_ss / static_cast<Scalar>(total_samples - 1)) : Scalar(0);

            return Stats{combined_mean, combined_std, total_samples};
        };

        Stats main_stat = average_stats(scores | std::views::transform([](const SelectionScore& s) { return s.main_metric; }));
        Stats tie_stat = average_stats(scores | std::views::transform([](const SelectionScore& s) { return s.tie_metric; }));
        return {main_stat, tie_stat};
    }

  private:
    // Returns the UCB of a given Stat
    OUT static Scalar ucb(Stats stats) noexcept { return stats.mean + 0.5 * stats.std; }
};