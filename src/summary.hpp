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

    void add_trial(const Metrics& run) {
        runs.push_back(run);
    }

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
        // A single run has no spread to speak of, so "± 0.000" would only read as measured agreement
        const auto figure = [this](const char* label, Scalar Metrics::* field, int decimals = 3, Scalar factor = 1, const char* unit = "") {
            const Stats stat = get(field);
            if (runs.size() > 1) {
                std::println("  • {}: {:.{}f}{} ± {:.{}f}{}", label, stat.mean * factor, decimals, unit, stat.std * factor, decimals, unit);
            } else {
                std::println("  • {}: {:.{}f}{}", label, stat.mean * factor, decimals, unit);
            }
        };

        if (track_accuracy) {
            figure("Accuracy", &Metrics::accuracy, 2, 100.0, "%");
            figure("Error", &Metrics::error);
            figure("MSE", &Metrics::mse);
        } else {
            figure("MSE", &Metrics::mse);
            figure("MEE", &Metrics::mee);
        }
        std::println("  ({} {})", runs.size(), runs.size() == 1 ? "run" : "runs");
    }
};

// Collects the best metric of each run grouped for the entire split
struct SplitSummary {
    RunSummary training;
    RunSummary holdout;

    void add_trial(const RunCurves& run) {
        training.add_trial(run.training.at(run.best_epoch));
        holdout.add_trial(run.holdout.at(run.best_epoch));
    }

    void add_trial(const Metrics& training_metrics, const Metrics& holdout_metrics) {
        training.add_trial(training_metrics);
        holdout.add_trial(holdout_metrics);
    }

    void print(bool track_accuracy, const std::string& holdout_name = "Holdout") const {
        std::println("\nSplit Summary:");
        std::println(" • Training Set:");
        training.print(track_accuracy);
        std::println(" • {} Set:", holdout_name);
        holdout.print(track_accuracy);
    }
};