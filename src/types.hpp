#pragma once

#include <Eigen/Dense>
#include <map>
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

enum class InitType { RANDOM,
                      LECUN,
                      GLOROT,
                      HE };

enum class DatasetType { XOR,
                         XOR_HOT,
                         MNIST };

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

const std::map<InitType, std::string> initialization_type_to_string = {
    {InitType::RANDOM, "Random"},
    {InitType::LECUN, "LeCun"},
    {InitType::GLOROT, "Glorot"},
    {InitType::HE, "He"},
};

const std::map<DatasetType, std::string> dataset_type_to_string = {
    {DatasetType::XOR, "XOR"},
    {DatasetType::XOR_HOT, "XOR_HOT"},
    {DatasetType::MNIST, "MNIST"},
};