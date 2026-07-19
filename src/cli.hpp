#pragma once

#include "dataset.hpp"
#include "types.hpp"

struct Args {
    DatasetType dataset_type;
    std::vector<int> net_struct;
    ActivationType hidden_activation;
    ActivationType output_activation;

    int epochs;
    int batch_size;
    Scalar eta;
    Scalar lambda;
    Scalar alpha;

    std::string log_file;
    std::string dump_file;
    std::string load_file;
};

Args parse_args(int argc, char* argv[]);