#pragma once

#include "types.hpp"

#include <Eigen/Dense>
#include <map>
#include <string>

using Scalar = float;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<Scalar, Eigen::Dynamic> Vector;

typedef std::function<Scalar(const Matrix&, const Matrix&)> LossFunction;
typedef std::function<Matrix(const Matrix&, const Matrix&)> LossDerivative;
typedef std::function<Matrix(const Matrix&)> ActivationFunction;

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

const std::map<ActivationType, std::string> activation_to_str = {
    {ActivationType::RELU, "ReLU"},
    {ActivationType::SIGMOID, "Sigmoid"},
    {ActivationType::TANH, "Tanh"},
    {ActivationType::LINEAR, "Linear"},
    {ActivationType::SOFTMAX, "Softmax"},
};

const std::map<LossType, std::string> loss_to_str = {
    {LossType::MSE, "Mean Squared Error"},
    {LossType::CCE, "Categorical Cross-Entropy"},
};

const std::map<TaskType, std::string> task_to_str = {
    {TaskType::REGRESSION, "Regression"},
    {TaskType::CLASSIFICATION, "Classification"},
};

const std::map<InitType, std::string> init_to_str = {
    {InitType::RANDOM, "Random"},
    {InitType::LECUN, "LeCun"},
    {InitType::GLOROT, "Glorot"},
    {InitType::HE, "He"},
};

const std::map<DatasetType, std::string> dataset_to_str = {
    {DatasetType::XOR, "XOR"},
    {DatasetType::XOR_HOT, "XOR_HOT"},
    {DatasetType::MNIST, "MNIST"},
};

const std::map<std::string, ActivationType> str_to_activation{
    {"sigmoid", ActivationType::SIGMOID},
    {"relu", ActivationType::RELU},
    {"tanh", ActivationType::TANH},
    {"softmax", ActivationType::SOFTMAX},
    {"linear", ActivationType::LINEAR}};

const std::map<std::string, InitType> str_to_init{
    {"random", InitType::RANDOM},
    {"lecun", InitType::LECUN},
    {"glorot", InitType::GLOROT},
    {"he", InitType::HE}};