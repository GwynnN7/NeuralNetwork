#pragma once

#include <Eigen/Dense>

using Scalar = double;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<Scalar, Eigen::Dynamic> Vector;

enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX // ONLY WITH CCE LOSS FUNCTION
};

enum class LossType { MSE,
                      CCE };
enum class TaskType { REGRESSION,
                      CLASSIFICATION };