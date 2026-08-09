#pragma once

#include "types.hpp"

struct Args {
    // Epochs parameters
    int epochs;
    Scalar patience;
    Scalar warmup;
    StoppingRule stopping_rule;

    // Dataset parameters
    DatasetType dataset_type;
    NormalizationType normalization_type;
    Scalar train_ratio;
    Scalar dataset_ratio;

    // Cross-validation parameters
    std::string grid_file;
    int trials;
    int inner_folds;
    int outer_folds;
    bool shuffle = false;

    // Configuration parameters
    std::string name;
    int seed;
    bool dump = false;
    bool train = false;

    static Args parse(int argc, char* argv[]);
};