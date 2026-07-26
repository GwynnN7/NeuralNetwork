#pragma once

#include <Eigen/Dense>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

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

const std::map<ActivationType, std::string> activation_type_to_string = {
    {ActivationType::RELU, "ReLU"},
    {ActivationType::SIGMOID, "Sigmoid"},
    {ActivationType::TANH, "Tanh"},
    {ActivationType::LINEAR, "Linear"},
    {ActivationType::SOFTMAX, "Softmax"},
};

enum class LossType { MSE,
                      CCE };
const std::map<LossType, std::string> loss_type_to_string = {
    {LossType::MSE, "Mean Squared Error"},
    {LossType::CCE, "Categorical Cross-Entropy"},
};

enum class TaskType { REGRESSION,
                      CLASSIFICATION };
const std::map<TaskType, std::string> task_type_to_string = {
    {TaskType::REGRESSION, "Regression"},
    {TaskType::CLASSIFICATION, "Classification"},
};

enum class InitializationType { RANDOM,
                                LECUN,
                                GLOROT,
                                HE };
const std::map<InitializationType, std::string> initialization_type_to_string = {
    {InitializationType::RANDOM, "Random"},
    {InitializationType::LECUN, "LeCun"},
    {InitializationType::GLOROT, "Glorot"},
    {InitializationType::HE, "He"},
};

inline int reverseInt(int i) {
    return __builtin_bswap32(i);
}

template <typename T>
inline std::string to_string_rounded(const T value, const int n = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << value;
    return out.str();
}