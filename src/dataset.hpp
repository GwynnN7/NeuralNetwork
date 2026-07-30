#pragma once

#include "types.hpp"

#include <map>
#include <print>

struct Dataset {
    DatasetType type;
    Matrix features, labels;
    int num_samples, num_features, num_classes;

    Dataset(DatasetType type, const Matrix& features, const Matrix& labels)
        : type(type), features(features), labels(labels) {

        num_samples = features.cols();
        num_features = features.rows();
        num_classes = labels.rows();

        std::println("\nDataset Info:");
        std::println("{:<25}{}", " • Type:", dataset_to_str.at(type));
        std::println("{:<25}{}", " • Samples:", num_samples);
        std::println("{:<25}{}", " • Features:", num_features);
        std::println("{:<25}{}", " • Classes:", num_classes);
    }
};

struct DataSplit {
    std::vector<int> train_indices;
    std::vector<int> test_indices;
};

Matrix load_csv(const std::string& filename);
Dataset load_dataset(DatasetType dataset_type, Scalar dataset_ratio);
std::vector<DataSplit> split_dataset(int num_samples, int k, Scalar train_ratio, bool shuffle = true);