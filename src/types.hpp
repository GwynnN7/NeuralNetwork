#pragma once

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <map>
#include <string>

// Use the NN_DOUBLE_PRECISION CMake option for double precision
#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#endif

// Define types alias for Scalar and commonly used Matrix, Vector and function types
using Scalar = SCALAR_TYPE;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

using LossFunction = std::function<Scalar(const Matrix&, const Matrix&)>;
using LossDerivative = std::function<Matrix(const Matrix&, const Matrix&)>;
using ActivationFunction = std::function<Matrix(const Matrix&)>;

// Define a small epsilon value used in optimizer denominators
inline constexpr Scalar EPSILON = 1e-8;
// Define a small epsilon value to avoid log(0) in loss functions (that works for both float and double, *8 is a safety for single precision)
inline constexpr Scalar LOSS_EPS = std::numeric_limits<Scalar>::epsilon() * 8;

// Define a relative tolerance for early stopping
inline constexpr Scalar ES_REL_TOL = 0.001;
// Define the frequency of logging metrics to a file during training
inline constexpr int LOG_FREQ = 25;
// Sliding window to score models during model selection
inline constexpr int SELECTION_WINDOW = 20;

// Define a multiplier for the target learning rate in linear decay
inline constexpr Scalar TARGET_ETA_MULTIPLIER = 0.01;
// Define a multiplier for the tau parameter in linear learning rate decay
inline constexpr Scalar TAU_MULTIPLIER = 0.8;

// Number of hyperparameter columns required in the grid-search CSV. 12th column (beta1) is optional and defaults to ADAM_B1
inline constexpr int HYPERPARAMS_NUM = 11;
inline constexpr int HYPERPARAMS_NUM_OPT = 12;

// Default beta1/beta2 parameters for the Adam optimizer
inline constexpr Scalar ADAM_B1 = 0.9;
inline constexpr Scalar ADAM_B2 = 0.999;

enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX
};

enum class OptimizerType {
    SGD,
    RMSPROP,
    ADAM
};

enum class LossType { MSE,
                      BCE,
                      CCE };

enum class TaskType { REGRESSION,
                      CLASSIFICATION };

enum class InitType { RANDOM,
                      LECUN,
                      GLOROT,
                      HE };

enum class DatasetType { XOR,
                         XOR_HOT,
                         MONK_1,
                         MONK_1_HOT,
                         MONK_2,
                         MONK_2_HOT,
                         MONK_3,
                         MONK_3_HOT,
                         MNIST };

namespace Maps {
const std::map<ActivationType, std::string> activation_to_str = {
    {ActivationType::RELU, "ReLU"},
    {ActivationType::SIGMOID, "Sigmoid"},
    {ActivationType::TANH, "Tanh"},
    {ActivationType::LINEAR, "Linear"},
    {ActivationType::SOFTMAX, "Softmax"},
};

const std::map<LossType, std::string> loss_to_str = {
    {LossType::MSE, "Mean Squared Error"},
    {LossType::BCE, "Binary Cross-Entropy"},
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

const std::map<OptimizerType, std::string> optimizer_to_str = {
    {OptimizerType::SGD, "SGD"},
    {OptimizerType::ADAM, "Adam"},
    {OptimizerType::RMSPROP, "RMSProp"},
};

const std::map<DatasetType, std::string> dataset_to_str = {
    {DatasetType::XOR, "XOR"},
    {DatasetType::XOR_HOT, "XOR_HOT"},
    {DatasetType::MONK_1, "MONK_1"},
    {DatasetType::MONK_1_HOT, "MONK_1_HOT"},
    {DatasetType::MONK_2, "MONK_2"},
    {DatasetType::MONK_2_HOT, "MONK_2_HOT"},
    {DatasetType::MONK_3, "MONK_3"},
    {DatasetType::MONK_3_HOT, "MONK_3_HOT"},
    {DatasetType::MNIST, "MNIST"},
};

const std::map<std::string, DatasetType> str_to_dataset{
    {"xor", DatasetType::XOR},
    {"xor_hot", DatasetType::XOR_HOT},
    {"monk_1", DatasetType::MONK_1},
    {"monk_1_hot", DatasetType::MONK_1_HOT},
    {"monk_2", DatasetType::MONK_2},
    {"monk_2_hot", DatasetType::MONK_2_HOT},
    {"monk_3", DatasetType::MONK_3},
    {"monk_3_hot", DatasetType::MONK_3_HOT},
    {"mnist", DatasetType::MNIST},
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

const std::map<std::string, OptimizerType> str_to_optimizer{
    {"sgd", OptimizerType::SGD},
    {"adam", OptimizerType::ADAM},
    {"rmsprop", OptimizerType::RMSPROP}};

const std::map<std::string, LossType> str_to_loss{
    {"mse", LossType::MSE},
    {"bce", LossType::BCE},
    {"cce", LossType::CCE}};
} // namespace Maps