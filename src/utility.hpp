#pragma once

#include "network.hpp"
#include "types.hpp"

#include <csignal>
#include <iomanip>
#include <iostream>
#include <map>
#include <print>
#include <random>

inline std::string MODEL_PATH;

inline volatile std::sig_atomic_t early_stop_flag = 0;
inline void handle_sigint(int sig) {
    if (early_stop_flag) {
        std::exit(sig);
    }
    early_stop_flag = 1;
}

inline std::mt19937& get_random_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

inline void set_random_seed(unsigned int seed) {
    get_random_generator().seed(seed);
}

inline void print_model(const Model& model) {
    std::println("\nTraining Configuration:");

    std::println(" • {:<25}{}", "Batch Size:", model.batch_size);
    std::println(" • {:<25}{}", "Learning Rate:", model.eta);
    std::println(" • {:<25}{}", "Regularization:", model.lambda);
    std::println(" • {:<25}{}", "Momentum:", model.alpha);
    std::println(" • {:<25}{}", "Hidden Activation:", activation_to_str.at(model.hidden_activation));
    std::println(" • {:<25}{}", "Output Activation:", activation_to_str.at(model.output_activation));
    std::println(" • {:<25}{}", "Weight Init:", init_to_str.at(model.init_type));
}

inline void print_random_samples(Matrix& target, Matrix& predictions) {
    std::println("\nRandom Samples Predictions:");

    std::uniform_int_distribution<int> dist(0, target.cols() - 1);
    for (int j = 0; j < std::min(3, (int)target.cols()); ++j) {
        int sample_index = dist(get_random_generator());
        if (j > 0) {
            std::println("   --------------{}", std::string(target.rows() * 6 - 1, '-'));
        }
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " • Target Label: " << target.col(sample_index).transpose() << "\n";
        std::cout << " • Prediction:   " << predictions.col(sample_index).transpose() << "\n";
    }
}