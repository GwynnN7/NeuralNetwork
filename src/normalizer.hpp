#pragma once

#include "types.hpp"

#include <algorithm>
#include <optional>
#include <utility>

class Normalizer {
  public:
    Normalizer() = default;

    OUT static Normalizer fit(NormalizationType type, const Matrix& training_data) {
        Normalizer normalizer;
        if (type == NormalizationType::NONE || training_data.cols() == 0) {
            return normalizer;
        }
        const Eigen::Index features = training_data.rows();
        normalizer.offset = Vector::Zero(features);
        normalizer.scaling = Vector::Ones(features);

        switch (type) {
        case NormalizationType::MIN_MAX: {
            normalizer.offset = training_data.rowwise().minCoeff();
            normalizer.scaling = training_data.rowwise().maxCoeff() - normalizer.offset;
            break;
        }
        case NormalizationType::ABS_MAX: {
            normalizer.scaling = training_data.cwiseAbs().rowwise().maxCoeff();
            break;
        }
        case NormalizationType::Z_SCORE: {
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
        return (data.colwise() - offset).array().colwise() / scaling.array();
    }

    OUT Matrix revert(const Matrix& data) const {
        if (!active()) {
            return data;
        }
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

    void load(Vector loaded_offset, Vector loaded_scaling) {
        offset = std::move(loaded_offset);
        scaling = std::move(loaded_scaling);
    }

  private:
    OUT bool active() const noexcept { return offset.size() > 0; }

    Vector offset, scaling;
};
