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
    SOFTMAX
};

enum class LossType { MSE,
                      CCE };
enum class TaskType { REGRESSION,
                      CLASSIFICATION };

enum class InitializationType { RANDOM,
                                LECUN,
                                GLOROT,
                                HE };

inline int reverseInt(int i) {
    return __builtin_bswap32(i);
}