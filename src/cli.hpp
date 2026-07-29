#pragma once

#include "types.hpp"

struct Args {
    DatasetType dataset_type;
    std::vector<int> net_struct;
    ActivationType hidden_activation;
    ActivationType output_activation;
    InitType init_type;

    int epochs;
    int batch_size;
    Scalar eta;
    Scalar lambda;
    Scalar alpha;

    Scalar train_ratio;
    Scalar dataset_ratio;

    std::string name;
    bool dump = false;
    bool load = false;

    int seed;
    int k_folds;
    bool shuffle = false;
};

Args parse_args(int argc, char* argv[]);
void print_args(const Args& args);