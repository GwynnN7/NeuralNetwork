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

enum class LossType { MSE,
                      CCE };

enum class TaskType { REGRESSION,
                      CLASSIFICATION };

enum class InitializationType { RANDOM,
                                LECUN,
                                GLOROT,
                                HE };

const std::map<ActivationType, std::string> activation_type_to_string = {
    {ActivationType::RELU, "ReLU"},
    {ActivationType::SIGMOID, "Sigmoid"},
    {ActivationType::TANH, "Tanh"},
    {ActivationType::LINEAR, "Linear"},
    {ActivationType::SOFTMAX, "Softmax"},
};
const std::map<LossType, std::string> loss_type_to_string = {
    {LossType::MSE, "Mean Squared Error"},
    {LossType::CCE, "Categorical Cross-Entropy"},
};
const std::map<TaskType, std::string> task_type_to_string = {
    {TaskType::REGRESSION, "Regression"},
    {TaskType::CLASSIFICATION, "Classification"},
};
const std::map<InitializationType, std::string> initialization_type_to_string = {
    {InitializationType::RANDOM, "Random"},
    {InitializationType::LECUN, "LeCun"},
    {InitializationType::GLOROT, "Glorot"},
    {InitializationType::HE, "He"},
};

// Utility function to reverse the byte order of an integer (used for reading MNIST files)
inline int reverseInt(int i) {
    return __builtin_bswap32(i);
}

// Utility function to convert a rounded value to a string
template <typename T>
inline std::string to_string_rounded(const T value, const int n = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << value;
    return out.str();
}