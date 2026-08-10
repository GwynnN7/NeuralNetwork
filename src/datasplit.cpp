#include "datasplit.hpp"

#include "cli.hpp"
#include "dataset.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

// Generates a single holdout split
DataSplit Splitter::holdout_split(int num_samples, bool use_default) const {
    if (num_samples < 2) {
        throw std::invalid_argument("Need at least 2 samples to split the dataset");
    }
    if (!(args.train_ratio > 0) || args.train_ratio >= 1) {
        throw std::invalid_argument("Argument train_ratio must be in (0, 1)");
    }

    // Use the dataset's predefined number of training samples
    const bool use_default_split = use_default && dataset.train_samples && *dataset.train_samples > 0 && *dataset.train_samples < num_samples;

    // Create a vector of indices representing the samples
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices if not using default split
    if (args.shuffle && !use_default_split) {
        std::ranges::shuffle(indices, get_split_generator());
    }

    DataSplit holdout;
    int train_size = use_default_split ? *dataset.train_samples : static_cast<int>(num_samples * args.train_ratio);
    // Avoid empty sets by clamping the train_size
    train_size = std::clamp(train_size, 1, num_samples - 1);
    // Assign the first 'train_size' samples to the training set and the rest to the test set
    holdout.train_indices.assign(indices.begin(), indices.begin() + train_size);
    holdout.test_indices.assign(indices.begin() + train_size, indices.end());
    return holdout;
}

// Generates a k-fold split with the given indices
std::vector<DataSplit> Splitter::kfold(std::vector<int> indices, int k) const {
    if (k < 2) {
        throw std::invalid_argument("Need at least 2 folds for k-fold cross-validation");
    }
    // Shuffle the indices to ensure random distribution of samples across folds
    if (args.shuffle) {
        std::ranges::shuffle(indices, get_split_generator());
    }

    const int num_samples = static_cast<int>(indices.size());
    if (num_samples < 2) {
        throw std::invalid_argument("Need at least 2 samples to split the dataset");
    }
    if (k > num_samples) {
        throw std::invalid_argument(std::format("Cannot split {} samples into {} folds", num_samples, k));
    }

    std::vector<DataSplit> folds(k);
    // Calculate the size of each fold and distribute the remaining samples on the first few folds
    const int fold_size = num_samples / k;
    const int remaining_samples = num_samples % k;

    int current_start = 0;
    for (int i = 0; i < k; ++i) {
        // Add one more sample to each fold until the remaining samples are distributed
        const int current_fold_size = fold_size + (i < remaining_samples ? 1 : 0);
        const int end = current_start + current_fold_size;

        // Assign test set
        folds[i].test_indices.assign(indices.begin() + current_start, indices.begin() + end);

        // Assign train set (samples before + samples after)
        folds[i].train_indices.assign(indices.begin(), indices.begin() + current_start);
        folds[i].train_indices.insert(folds[i].train_indices.end(), indices.begin() + end, indices.end());

        // Move the sliding window forward
        current_start = end;
    }

    return folds;
}

// Generates a standard k-fold split (without considering class distribution)
std::vector<DataSplit> Splitter::standard_split(int k, int num_samples) const {
    // Create a vector of indices representing the samples
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    return kfold(std::move(indices), k);
}

// Generates a stratified k-fold split, so each fold has approximately the same class distribution
std::vector<DataSplit> Splitter::stratified_split(int k, const Matrix& labels) const {
    const int num_samples = static_cast<int>(labels.cols());
    const int num_classes = static_cast<int>(labels.rows());

    // Create a vector of indices per class
    // For binary classification there are two classes (0 and 1), while for one_hot there are `num_classes` classes
    std::vector<std::vector<int>> class_indices(num_classes == 1 ? 2 : num_classes);

    // Group indices by class
    for (int i = 0; i < num_samples; ++i) {
        class_indices[get_task_class(labels, i)].push_back(i);
    }

    // Split each class over the same folds, so every fold keeps the dataset's class proportions
    std::vector<DataSplit> folds(k);
    for (const auto& indices : class_indices) {
        if (static_cast<int>(indices.size()) < k) {
            throw std::invalid_argument(std::format("{} samples are too few for {} stratified folds", indices.size(), k));
        }
        // Generate a k-fold split for the current class and concatenate it to the main folds
        for (auto&& [fold, class_fold] : std::views::zip(folds, kfold(indices, k))) {
            fold.concat(class_fold);
        }
    }

    return folds;
}

// Determines and generates a split of the dataset or of the provided indices
std::vector<DataSplit> Splitter::get(int folds, const std::vector<int>* indices) const {
    const int samples = (indices != nullptr) ? static_cast<int>(indices->size()) : dataset.num_samples;
    std::vector<DataSplit> splits;

    // Determine the type of split based on the number of folds and the dataset's task type
    if (folds == 1) {
        splits = {holdout_split(samples, indices == nullptr)};
    } else if (dataset.task == TaskType::CLASSIFICATION) {
        splits = (indices != nullptr) ? stratified_split(folds, dataset.labels(Eigen::placeholders::all, *indices))
                                      : stratified_split(folds, dataset.labels);
    } else {
        splits = standard_split(folds, samples);
    }

    // Remap the indices of the splits to the original dataset indices
    if (indices != nullptr) {
        for (DataSplit& split : splits) {
            for (int& idx : split.train_indices) {
                idx = indices->at(idx);
            }
            for (int& idx : split.test_indices) {
                idx = indices->at(idx);
            }
        }
    }
    return splits;
}
