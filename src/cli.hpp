#pragma once

#include "types.hpp"

struct Args {
    DatasetType dataset_type;
    std::string name;

    int epochs;
    Scalar train_ratio;
    Scalar dataset_ratio;

    std::string model_file;

    int seed;
    int inner_folds;
    int outer_folds;
    bool shuffle = false;

    bool dump = false;
    bool train = false;
};

Args parse_args(int argc, char* argv[]);