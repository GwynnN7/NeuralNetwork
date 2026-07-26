#pragma once

#include "types.hpp"

#include <iomanip>
#include <iostream>
#include <map>

enum class DatasetType { XOR,
                         XOR_HOT,
                         MNIST };

const std::map<DatasetType, std::string> dataset_type_to_string = {
    {DatasetType::XOR, "XOR"},
    {DatasetType::XOR_HOT, "XOR_HOT"},
    {DatasetType::MNIST, "MNIST"},
};

struct Dataset {
    DatasetType type;
    Matrix features;
    Matrix labels;
    int num_samples;
    int num_features;
    int num_classes;

    Dataset(DatasetType type, const Matrix& features, const Matrix& labels) : type(type), features(features), labels(labels) {
        num_samples = features.cols();
        num_features = features.rows();
        num_classes = labels.rows();
    }
};

struct ModelSet {
    Dataset train_set;
    Dataset test_set;

    ModelSet(const Dataset& train_set, const Dataset& test_set) : train_set(train_set), test_set(test_set) {
        std::cout << std::endl
                  << "Dataset Info:" << "\n"
                  << std::left << std::setw(25) << " • Type: " << dataset_type_to_string.at(train_set.type) << "\n"
                  << std::left << std::setw(25) << " • Samples: " << train_set.num_samples << "  |  " << test_set.num_samples << "\n"
                  << std::left << std::setw(25) << " • Features:" << train_set.num_features << "  |  " << test_set.num_features << "\n"
                  << std::left << std::setw(25) << " • Classes:" << train_set.num_classes << "  |  " << test_set.num_classes << "\n";
    }
};

struct SetIndices {
    std::vector<int> train_indices;
    std::vector<int> test_indices;
};

Matrix load_csv(const std::string& filename);
ModelSet load_dataset(DatasetType dataset_type, Scalar train_ratio, Scalar dataset_ratio);