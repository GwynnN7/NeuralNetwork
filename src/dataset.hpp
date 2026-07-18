#pragma once

#include "types.hpp"

struct Dataset {
    Matrix features;
    Matrix labels;
    int num_samples;
    int num_features;
    int num_classes;
};

enum class DatasetType { XOR, XOR_HOT, MNIST };

Matrix load_csv(const std::string &filename);
Dataset load_dataset(DatasetType dataset_type);