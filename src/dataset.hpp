#pragma once

#include "types.hpp"

#include <print>

struct Dataset {
    DatasetType type;
    TaskType task;
    Matrix features, labels;
    int num_samples, num_features, num_classes;

    Dataset(DatasetType type, TaskType task, const Matrix& features, const Matrix& labels)
        : type(type), task(task), features(features), labels(labels) {

        num_samples = features.cols();
        num_features = features.rows();
        num_classes = labels.rows();

        std::println("\nDataset Info:");
        std::println("{:<25}{}", " • Type:", Maps::dataset_to_str.at(type));
        std::println("{:<25}{}", " • Task:", Maps::task_to_str.at(task));
        std::println("{:<25}{}", " • Samples:", num_samples);
        std::println("{:<25}{}", " • Features:", num_features);
        std::println("{:<25}{}", " • Classes:", num_classes);
    }

    static Dataset load(DatasetType type, Scalar dataset_ratio);
};

struct DataSplit {
    std::vector<int> train_indices;
    std::vector<int> test_indices;

    static std::vector<DataSplit> split(int num_samples, int k, Scalar train_ratio, bool shuffle = true);
};