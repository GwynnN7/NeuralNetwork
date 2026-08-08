#pragma once

#include "metrics.hpp"
#include "types.hpp"

#include <cmath>
#include <numeric>
#include <print>
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

    bool empty() const { return runs.empty(); }
    size_t size() const { return runs.size(); }

    void add_trial(const Metrics& run) {
        runs.push_back(run);
    }

    Stats get(Scalar Metrics::* field) const {
        if (runs.empty()) {
            return {};
        }
        const Scalar n = static_cast<Scalar>(runs.size());
        const Scalar mean = std::accumulate(runs.begin(), runs.end(), Scalar(0), [field](Scalar acc, const Metrics& metric) { return acc + metric.*field; }) / n;
        if (runs.size() < 2) {
            return {mean, 0};
        }
        const Scalar var = std::accumulate(runs.begin(), runs.end(), Scalar(0), [field, mean](Scalar acc, const Metrics& metric) { return acc + (metric.*field - mean) * (metric.*field - mean); });
        return {mean, std::sqrt(var / (n - 1))};
    }

    void print(bool track_accuracy) const {
        if (runs.empty()) {
            std::println(" • (no runs recorded)");
            return;
        }
        if (track_accuracy) {
            const Stats acc = get(&Metrics::accuracy);
            std::println(" • Accuracy: {:.2f}%±{:.2f}%", acc.mean * 100.0, acc.std * 100.0);
        }
        const Stats err = get(&Metrics::error);
        const Stats mse = get(&Metrics::mse);
        std::println(" • Error: {:.3f}±{:.3f} - MSE: {:.3f}±{:.3f} ({} runs)", err.mean, err.std, mse.mean, mse.std, runs.size());
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