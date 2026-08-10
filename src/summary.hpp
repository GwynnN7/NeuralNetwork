#pragma once

#include "metrics.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <print>
#include <ranges>
#include <string>
#include <vector>

// Mean and sample standard deviation
struct Stats {
    Scalar mean = 0;
    Scalar std = 0;
};

// Collects the best metric of each run
struct RunSummary {
    std::vector<Metrics> runs;

    OUT bool empty() const noexcept { return runs.empty(); }
    OUT size_t size() const noexcept { return runs.size(); }

    void add_run(const Metrics& run) {
        runs.push_back(run);
    }

    // Get the mean and standard deviation of a metric field across all runs
    OUT Stats get(Scalar Metrics::* field) const {
        if (empty()) {
            return {};
        }
        const Scalar n = static_cast<Scalar>(size());
        const auto values = runs | std::views::transform([field](const Metrics& metric) { return metric.*field; });
        const Scalar mean = std::ranges::fold_left(values, Scalar(0), std::plus{}) / n;
        if (size() < 2) {
            return {mean, 0};
        }
        const Scalar var = std::ranges::fold_left(values, Scalar(0), [mean](Scalar acc, Scalar value) { return acc + (value - mean) * (value - mean); });
        return {mean, std::sqrt(var / (n - 1))};
    }

    void print(bool track_accuracy) const {
        if (empty()) {
            std::println("  • (no runs recorded)");
            return;
        }
        const auto log = [this](const char* label, Scalar Metrics::* field, int decimals = 3, Scalar factor = 1, const char* unit = "") {
            const Stats stat = get(field);
            if (runs.size() > 1) {
                std::println("  • {}: {:.{}f}{} ± {:.{}f}{}", label, stat.mean * factor, decimals, unit, stat.std * factor, decimals, unit);
            } else {
                std::println("  • {}: {:.{}f}{}", label, stat.mean * factor, decimals, unit);
            }
        };

        if (track_accuracy) {
            log("Accuracy", &Metrics::accuracy, 2, 100.0, "%");
            log("Error", &Metrics::error);
            log("MSE", &Metrics::mse);
        } else {
            log("MEE", &Metrics::mee);
            log("MSE", &Metrics::mse);
        }
        std::println("  ({} {})", size(), size() == 1 ? "run" : "runs");
    }
};

// Collects the best metric of each run for both training and holdout sets
struct SplitSummary {
    RunSummary training;
    RunSummary holdout;

    // If an entire run is provided, add the best metrics from both training and holdout sets
    void add_run(const RunCurves& run) {
        // Don't add runs that were interrupted by an inf/NaN error or that have no epochs recorded
        if (run.training.invalid || run.training.empty()) {
            return;
        }
        training.add_run(run.training.at(run.best_epoch));
        holdout.add_run(run.holdout.at(run.best_epoch));
    }

    // If only one metric is provided, add it as a run
    void add_run(const Metrics& training_metrics, const Metrics& holdout_metrics) {
        training.add_run(training_metrics);
        holdout.add_run(holdout_metrics);
    }

    void print(bool track_accuracy, const std::string& holdout_name = "Holdout") const {
        std::println("\nSplit Summary:");
        std::println(" • Training Set:");
        training.print(track_accuracy);
        std::println(" • {} Set:", holdout_name);
        holdout.print(track_accuracy);
    }
};