#pragma once

#include "types.hpp"

#include <utility>

struct Dataset {
    DatasetType type;
    TaskType task;
    Matrix features, labels;
    int num_samples, num_features, num_classes;
    std::optional<int> train_samples; // Store the original number of training samples for datasets that have a predefined train/test split

    Dataset(DatasetType type, TaskType task, Matrix features, Matrix labels, std::optional<int> train_samples = std::nullopt)
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
    QUERY static Dataset load(DatasetType type, Scalar dataset_ratio);
};