#pragma once

#include "types.hpp"

#include <optional>
#include <utility>
#include <vector>

struct Dataset {
    DatasetType type;
    TaskType task;
    Matrix features, labels;
    int num_samples, num_features, num_classes;
    int train_samples; // Store the original number of training samples for datasets that have a predefined train/test split

    Dataset(DatasetType type, TaskType task, Matrix features, Matrix labels, int train_samples = 0)
        : type(type), task(task), features(std::move(features)), labels(std::move(labels)), train_samples(train_samples) {

        num_samples = static_cast<int>(this->features.cols());
        num_features = static_cast<int>(this->features.rows());
        num_classes = static_cast<int>(this->labels.rows());

        if (this->labels.cols() != this->features.cols()) {
            throw std::invalid_argument("Number of samples in features and labels must match");
        }
        if (num_samples == 0 || num_features == 0 || num_classes == 0) {
            throw std::invalid_argument("Dataset is empty");
        }
    }

    void print_info() const;
    static Dataset load(DatasetType type, Scalar dataset_ratio);
};

struct DataSplit {
    std::vector<int> train_indices;
    std::vector<int> test_indices;

    static std::vector<DataSplit> split(int k, Scalar train_ratio, int train_samples, int num_samples, bool shuffle, std::optional<unsigned int> seed = std::nullopt);
};