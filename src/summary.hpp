#pragma once

#include "metrics.hpp"
#include "types.hpp"

#include <algorithm>
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

    // Mean and std of a range of values
    template <std::ranges::input_range R>
    OUT static Stats of(R&& values) {
        const auto count = std::ranges::distance(values);
        if (count == 0) {
            return {};
        }
        const Scalar mean = std::ranges::fold_left(values, Scalar(0), std::plus{}) / static_cast<Scalar>(count);
        if (count < 2) {
            return {mean, 0};
        }
        const Scalar var = std::ranges::fold_left(values, Scalar(0), [mean](Scalar acc, Scalar value) { return acc + (value - mean) * (value - mean); });
        return {mean, std::sqrt(var / static_cast<Scalar>(count - 1))};
    }
};

// Collects the best metric of each run
struct RunSummary {
    std::vector<Metrics> runs; // Best metrics of each run

    OUT bool empty() const noexcept { return runs.empty(); }
    OUT size_t size() const noexcept { return runs.size(); }

    // Add the metrics of a run to the summary
    void add_run(const Metrics& run) {
        runs.push_back(run);
    }

    // Add the metrics of another RunSummary to this one
    void concat(const RunSummary& other) {
        runs.insert(runs.end(), other.runs.begin(), other.runs.end());
    }

    // Add the averaged metrics of another RunSummary to this one
    void add(const RunSummary& other) {
        Metrics mean{
            .mse = other.get(&Metrics::mse).mean,
            .mee = other.get(&Metrics::mee).mean,
            .error = other.get(&Metrics::error).mean,
            .accuracy = other.get(&Metrics::accuracy).mean,
            .weight = 1};
        runs.push_back(mean);
    }

    // Get the stats of a metric field across all runs
    OUT Stats get(Scalar Metrics::* field) const {
        return Stats::of(runs | std::views::transform([field](const Metrics& metric) { return metric.*field; }));
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
    }
};

// Collects the duration of each run
struct TimingSummary {
    std::vector<Scalar> seconds; // Duration of each run
    std::vector<int> epochs;     // Epochs of each run

    OUT bool empty() const noexcept { return seconds.empty(); }

    // Add duration and epochs of a run
    void add_run(Scalar duration, int num_epochs) {
        seconds.push_back(duration);
        epochs.push_back(num_epochs);
    }

    // Timing Stats per run
    OUT Stats per_run() const {
        return Stats::of(seconds);
    }

    // Concatenate the timing of another TimingSummary to this one
    void concat(const TimingSummary& other) {
        seconds.insert(seconds.end(), other.seconds.begin(), other.seconds.end());
        epochs.insert(epochs.end(), other.epochs.begin(), other.epochs.end());
    }

    // Average number of epochs a run lasted, which early stopping makes vary
    OUT int average_epochs() const {
        return empty() ? 0 : static_cast<int>(std::ranges::fold_left(epochs, 0, std::plus{}) / static_cast<Scalar>(epochs.size()));
    }

    // Average time per epoch across all runs in milliseconds
    OUT Scalar per_epoch_ms() const {
        const int total_epochs = std::ranges::fold_left(epochs, 0, std::plus{});
        if (total_epochs == 0) {
            return 0;
        }
        return total() * Scalar(1000) / static_cast<Scalar>(total_epochs);
    }

    // Total time of all runs in seconds
    OUT Scalar total() const {
        return std::ranges::fold_left(seconds, Scalar(0), std::plus{});
    }

    void print(const char* label = "Timing") const {
        if (empty()) {
            return;
        }
        const Stats run = per_run();
        std::println(" • {}:", label);
        std::println("  • ~Epochs: {} ({:.2f}ms each)", average_epochs(), per_epoch_ms());
        if (seconds.size() > 1) {
            std::println("  • Run: {:.2f}s±{:.2f}s", run.mean, run.std);
        } else {
            std::println("  • Run: {:.2f}s", run.mean);
        }
        std::println("  • Total: {:.2f}s", total());
    }
};

// Collects the best metric of each run for both training and holdout sets
struct SplitSummary {
    RunSummary training;
    RunSummary holdout;
    TimingSummary timing;

    // If an entire run is provided, add the best metrics from both training and holdout sets
    void add_run(const RunCurves& run) {
        // Don't add runs that were interrupted by an inf/NaN error or that have no epochs
        if (run.training.invalid || run.training.empty()) {
            return;
        }
        training.add_run(run.training.at(run.best_epoch));
        holdout.add_run(run.holdout.at(run.best_epoch));
        timing.add_run(run.duration, static_cast<int>(run.training.size()));
    }

    // If only one metric is provided, add it as a run
    void add_run(const Metrics& training_metrics, const Metrics& holdout_metrics) {
        training.add_run(training_metrics);
        holdout.add_run(holdout_metrics);
    }

    void print(bool track_accuracy, const std::string& title, const std::string& holdout_name) const {
        std::println("\n[{}, {} run(s)]", title, training.size());
        std::println(" • Training Set:");
        training.print(track_accuracy);
        std::println(" • {} Set:", holdout_name);
        holdout.print(track_accuracy);
        timing.print();
    }
};