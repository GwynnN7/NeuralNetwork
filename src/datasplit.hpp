#pragma once

#include "types.hpp"

#include <vector>

// A partition of sample indices into a part to train on and a part held out from training
struct DataSplit {
    std::vector<int> train_indices;
    std::vector<int> test_indices;

    void concat(const DataSplit& other) {
        train_indices.insert(train_indices.end(), other.train_indices.begin(), other.train_indices.end());
        test_indices.insert(test_indices.end(), other.test_indices.begin(), other.test_indices.end());
    }
};

// Forward declaration
struct Args;
struct Dataset;

// Splits builder for a dataset/args combination
class Splitter {
  public:
    Splitter(const Dataset& dataset, const Args& args) : dataset(dataset), args(args) {}

    QUERY std::vector<DataSplit> get(int folds, const std::vector<int>* indices = nullptr) const;
    QUERY DataSplit final_split() const;

  private:
    const Dataset& dataset;
    const Args& args;

    // `use_default` to try using the dataset's default train/test splitting
    QUERY DataSplit holdout_split(int num_samples, bool use_default) const;
    QUERY std::vector<DataSplit> standard_split(int k, int num_samples) const;
    QUERY std::vector<DataSplit> stratified_split(int k, const Matrix& labels) const;
    QUERY std::vector<DataSplit> kfold(std::vector<int> indices, int k) const;
};
