#pragma once

#include "types.hpp"

#include <algorithm>
#include <optional>
#include <utility>

/*
Normalization is fitted on the training set and applied to both training and test sets

MIN_MAX:
  Scale each feature to [0, 1] by subtracting the minimum and dividing by the range (max - min)
  x' = (x - min) / (max - min)
  Sensitive to outliers but good for data that is already bounded

ABS_MAX:
  Scale each feature to [-1, 1] by dividing by the absolute maximum value
  x' = x / max(|x|)
  Good for sparse data (with many zero values) because it doesn't subtract the mean, keeping the zeros
  By preserving sparsity, it saves memory and increases efficiency for sparse datasets

Z_SCORE:
  Standardize each feature to have zero mean and unit variance by subtracting the mean and dividing by the standard deviation
  x' = (x - mean) / std
  Less sensitive to outliers than min-max scaling, but still affected by extreme values

NONE:
  No normalization is applied, and the data is used as-is
  Useful for datasets that are already preprocessed or standardized
*/

class Normalizer {
  public:
    Normalizer() = default;

    // Fit a normalizer to the training data with a specified normalization type
    OUT static Normalizer fit(NormalizationType type, const Matrix& training_data) {
        Normalizer normalizer;
        if (type == NormalizationType::NONE || training_data.cols() == 0) {
            return normalizer;
        }
        const Eigen::Index features = training_data.rows();
        // Initialize offset and scaling vectors
        normalizer.offset = Vector::Zero(features);
        normalizer.scaling = Vector::Ones(features);

        switch (type) {
        case NormalizationType::MIN_MAX: {
            // x' = (x - min) / (max - min)
            normalizer.offset = training_data.rowwise().minCoeff();
            normalizer.scaling = training_data.rowwise().maxCoeff() - normalizer.offset;
            break;
        }
        case NormalizationType::ABS_MAX: {
            // x' = x / max(|x|)
            normalizer.scaling = training_data.cwiseAbs().rowwise().maxCoeff();
            break;
        }
        case NormalizationType::Z_SCORE: {
            // x' = (x - mean) / std
            normalizer.offset = training_data.rowwise().mean();
            const Eigen::Index divisor = std::max<Eigen::Index>(1, training_data.cols() - 1);
            normalizer.scaling = ((training_data.colwise() - normalizer.offset).rowwise().squaredNorm() / divisor).cwiseSqrt();
            break;
        }
        case NormalizationType::NONE:
            break;
        }

        // Avoid division by zero
        normalizer.scaling = (normalizer.scaling.array() != Scalar(0)).select(normalizer.scaling, Scalar(1));
        return normalizer;
    }

    OUT Matrix apply(const Matrix& data) const {
        if (!active()) {
            return data;
        }
        // Apply the normalization to the data using the stored offset and scaling
        // x' = (x - offset) / scaling
        return (data.colwise() - offset).array().colwise() / scaling.array();
    }

    OUT Matrix revert(const Matrix& data) const {
        if (!active()) {
            return data;
        }
        // Revert the normalization to get back the original data
        // x = (x' * scaling) + offset
        return (data.array().colwise() * scaling.array()).matrix().colwise() + offset;
    }

    OUT std::optional<std::pair<Eigen::Index, Eigen::Index>> size() const noexcept {
        if (!active()) {
            return std::nullopt;
        }
        return std::make_pair(offset.size(), scaling.size());
    }

    OUT const Scalar* offsets() const noexcept { return offset.data(); }
    OUT const Scalar* scalings() const noexcept { return scaling.data(); }

    // Load the offset and scaling vectors from dumped models
    void load(Vector loaded_offset, Vector loaded_scaling) {
        offset = std::move(loaded_offset);
        scaling = std::move(loaded_scaling);
    }

  private:
    OUT bool active() const noexcept { return offset.size() > 0; }
    Vector offset, scaling;
};
