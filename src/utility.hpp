#pragma once

#include "types.hpp"

#include <charconv>
#include <csignal>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Root directory for artifacts
inline std::string MODEL_PATH;

// Flag and signal handler to handle cli early stopping.
inline volatile std::sig_atomic_t early_stop_flag = 0;
inline void handle_signal(int) {
    early_stop_flag = early_stop_flag ? 0 : 1;
}

// Shared generator for building outer/inner folds
inline std::mt19937& get_split_generator() {
    static std::mt19937 split_generator(std::random_device{}());
    return split_generator;
}

// Shared generator for weight initialization and shuffling
inline std::mt19937& get_trial_generator() {
    static std::mt19937 trial_generator(std::random_device{}());
    return trial_generator;
}

// Set the random seed for reproducibility, mixing trial and fold indices for per-run reproducibility
inline void set_trial_seed(unsigned int seed, int trial, int fold) {
    // Mix the seed with trial and fold indices for per-run reproducibility
    std::seed_seq seq{seed, static_cast<unsigned int>(trial), static_cast<unsigned int>(fold)};
    get_trial_generator().seed(seq);
}

// Set the random seed for reproducibility
inline void set_split_seed(unsigned int seed) {
    get_split_generator().seed(seed);
}

// Get the class index for a given sample from the labels
inline int get_task_class(const Matrix& labels, const Eigen::Index sample_index) {
    if (labels.rows() == 1) { // Binary classification
        return labels(0, sample_index) >= Scalar(0.5) ? 1 : 0;
    } else { // Multi-class classification
        Eigen::Index class_index;
        labels.col(sample_index).maxCoeff(&class_index);
        return static_cast<int>(class_index);
    }
}

// Trim leading and trailing whitespace from a string
inline std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

// Parse a whole string as a number of type T
template <typename T>
T parse_number(const std::string& s) {
    T value{};
    const char* last = s.data() + s.size();
    const auto [ptr, err] = std::from_chars(s.data(), last, value);
    // If the error isn't none or the pointer didn't reach the end of the string
    if (err != std::errc{} || ptr != last) {
        throw std::runtime_error("'" + s + "' is not a valid number");
    }
    return value;
}

// Check if the whole string is a number ("-1" and "1e-3" count, "data_1" does not)
inline bool is_number(const std::string& s) {
    try {
        parse_number<double>(s);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Parse a cell, reporting the file and line it came from, and the field name if available
template <typename T>
T parse_cell(const std::string& cell, const std::string& filename, int line, const std::string& field = "") {
    try {
        return parse_number<T>(cell);
    } catch (const std::exception&) {
        throw std::runtime_error(filename + " line " + std::to_string(line) + ": '" + cell + "' is not a valid number" + (field.empty() ? "" : " for " + field));
    }
}

// Structure to hold a row of CSV with its line number
struct CsvRow {
    int line = 0;
    std::vector<std::string> cells;
};

// Load a CSV file into a vector of CsvRow, skipping comments and empty lines
inline std::vector<CsvRow> load_csv(const std::string& filename, char delimiter = ',', bool skip_header = false) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<CsvRow> rows;
    std::string line;
    int line_number = 0;
    bool first_row = true;

    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines, lines that contain only whitespace, and '#' comments
        const size_t first_char = line.find_first_not_of(" \t\r");
        if (first_char == std::string::npos || line[first_char] == '#') {
            continue;
        }

        std::vector<std::string> cells;
        std::stringstream line_stream(line);
        std::string cell;
        if (delimiter == ' ') {
            // Split on whitespace for space-delimited files (ignores multiple spaces, tabs, leading/trailing whitespace)
            while (line_stream >> cell) {
                cells.push_back(cell);
            }
        } else {
            while (std::getline(line_stream, cell, delimiter)) {
                cells.push_back(trim(cell));
            }
        }
        if (cells.empty()) {
            continue;
        }

        // Skip the header row
        if (first_row) {
            first_row = false;
            if (skip_header && !is_number(cells[0])) {
                continue;
            }
        }

        rows.push_back({line_number, std::move(cells)});
    }

    if (rows.empty()) {
        throw std::runtime_error("No data rows found in file: " + filename);
    }
    return rows;
}