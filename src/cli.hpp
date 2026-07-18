#pragma once

#include "dataset.hpp"
#include "types.hpp"

struct Args {
    DatasetType dataset_type;
    int epochs;
    int batch_size;
    Scalar eta;
    Scalar lambda;
    Scalar alpha;
    TaskType task_type;
};

Args parse_args(int argc, char *argv[]);