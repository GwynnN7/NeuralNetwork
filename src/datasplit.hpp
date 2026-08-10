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

    OUT std::vector<DataSplit> get(int folds, const std::vector<int>* indices = nullptr) const;

  private:
    const Dataset& dataset;
    const Args& args;

    // Single holdout split, where the test set is either the default test split or a (random) subset of the dataset
    OUT DataSplit holdout_split(int num_samples, bool use_default) const;
    // Standard k-fold split, where each fold has approximately the same number of samples
    OUT std::vector<DataSplit> standard_split(int k, int num_samples) const;
    // Stratified k-fold split, where each fold has approximately the same class distribution
    OUT std::vector<DataSplit> stratified_split(int k, const Matrix& labels) const;
    // Generates a k-fold split from a given set of indices
    OUT std::vector<DataSplit> kfold(std::vector<int> indices, int k) const;
};
