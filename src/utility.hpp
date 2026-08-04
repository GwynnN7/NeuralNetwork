#pragma once

#include "types.hpp"

#include <algorithm>
#include <bit>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <print>
#include <random>

inline std::string MODEL_PATH;

// Flag and signal handler to handle cli early stopping
inline volatile std::sig_atomic_t early_stop_flag = 0;
inline void handle_signal([[maybe_unused]] int sig) {
    early_stop_flag = early_stop_flag ? 0 : 1;
}

inline std::mt19937& get_random_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

inline void set_random_seed(unsigned int seed) {
    get_random_generator().seed(seed);
}

inline int swap_endian(int value) {
    if constexpr (std::endian::native == std::endian::little) {
        return std::byteswap(value);
    }
    return value;
}

inline void print_random_samples(Matrix& target, Matrix& predictions) {
    std::println("\nRandom Samples Predictions:");

    // Randomly select 3 samples from the dataset to display predictions
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