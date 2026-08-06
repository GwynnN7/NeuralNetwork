#pragma once

#include <csignal>
#include <random>
#include <string>

// Root directory for artifacts
inline std::string MODEL_PATH;

// Flag and signal handler to handle cli early stopping.
inline volatile std::sig_atomic_t early_stop_flag = 0;
inline void handle_signal([[maybe_unused]] int sig) {
    early_stop_flag = early_stop_flag ? 0 : 1;
}

// Shared generator for weight initialization and shuffling
inline std::mt19937& get_random_generator() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

inline void set_random_seed(unsigned int seed) {
    get_random_generator().seed(seed);
}
