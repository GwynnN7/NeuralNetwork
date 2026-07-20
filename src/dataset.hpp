#pragma once

#include "types.hpp"

#include <iomanip>
#include <iostream>

struct Dataset {
    Matrix features;
    Matrix labels;
    int num_samples;
    int num_features;
    int num_classes;

    Dataset(const Matrix& features, const Matrix& labels) : features(features), labels(labels) {
        num_samples = features.cols();
        num_features = features.rows();
        num_classes = labels.rows();
    }
};

struct ModelSet {
    Dataset train_set;
    Dataset test_set;

    ModelSet(const Dataset& train_set, const Dataset& test_set) : train_set(train_set), test_set(test_set) {
        std::cout << "Dataset Info:" << "\n"
                  << std::left << std::setw(25) << " • Samples: " << train_set.num_samples << "  |  " << test_set.num_samples << "\n"
                  << std::left << std::setw(25) << " • Features:" << train_set.num_features << "  |  " << test_set.num_features << "\n"
                  << std::left << std::setw(25) << " • Classes:" << train_set.num_classes << "  |  " << test_set.num_classes << "\n\n";
    }
};

struct SetIndices {
    std::vector<int> train_indices;
    std::vector<int> test_indices;
};

enum class DatasetType { XOR,
                         XOR_HOT,
                         MNIST };

Matrix load_csv(const std::string& filename);
ModelSet load_dataset(DatasetType dataset_type, Scalar train_ratio, Scalar dataset_ratio);