#pragma once

#include "types.hpp"

struct Args {
    DatasetType dataset_type;
    std::string name;

    int epochs;
    int patience;
    Scalar train_ratio;
    Scalar dataset_ratio;

    std::string model_file;

    int seed;
    int inner_folds;
    int outer_folds;
    bool shuffle = false;

    bool dump = false;
    bool train = false;

    static Args parse(int argc, char* argv[]);
};