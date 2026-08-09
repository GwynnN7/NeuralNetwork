#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

// Use the NN_DOUBLE_PRECISION CMake option for double precision
#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#endif

// Define a macro for marking functions as [[nodiscard]]
#define OUT [[nodiscard]]

// Define types for Scalar and commonly used Matrix and Vector
using Scalar = SCALAR_TYPE;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

// Define types for Activation and Loss functions
using LossFunction = Scalar (*)(const Matrix&, const Matrix&);
using LossDerivative = Matrix (*)(const Matrix&, const Matrix&);
using ActivationFunction = Matrix (*)(const Matrix&);

struct ActivationPair {
    ActivationFunction function = nullptr;
    ActivationFunction derivative = nullptr;
};

struct LossPair {
    LossFunction function = nullptr;
    LossDerivative derivative = nullptr;
};

struct Parameters {
    Matrix W;
    Vector b;

    Parameters() : W(), b() {}
    Parameters(Eigen::Index output_size, Eigen::Index input_size) : W(Matrix::Zero(output_size, input_size)), b(Vector::Zero(output_size)) {}
    Parameters(Matrix weights, Vector biases) : W(std::move(weights)), b(std::move(biases)) {
        if (W.rows() != b.size()) {
            throw std::invalid_argument("Weights and biases must have compatible dimensions");
        }
    }
    void operator+=(const Parameters& other) {
        if (W.rows() != other.W.rows() || W.cols() != other.W.cols() || b.size() != other.b.size()) {
            throw std::invalid_argument("Cannot sum Parameters with different dimensions");
        }
        W += other.W;
        b += other.b;
    }
};

// Define a small epsilon value used in optimizer denominators
inline constexpr Scalar EPSILON = 1e-8;
// Define a small epsilon value to avoid log(0) in loss functions (that works for both float and double, *8 is a safety for single precision)
inline constexpr Scalar LOSS_EPSILON = std::numeric_limits<Scalar>::epsilon() * 8;
// Define a constant for infinity, used in metrics and selection scores
inline constexpr Scalar INF = std::numeric_limits<Scalar>::infinity();

// Define a relative tolerance for early stopping
inline constexpr Scalar ES_TOLERANCE = 0.001;
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

enum class StoppingRule {
    CONVERGENCE,
    ERROR_LEVEL
};

enum class NormalizationType {
    NONE,
    MIN_MAX,
    ABS_MAX,
    Z_SCORE
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
                         MONK1,
                         MONK1_HOT,
                         MONK2,
                         MONK2_HOT,
                         MONK3,
                         MONK3_HOT,
                         MLCUP,
                         MNIST };

// Lookup tables for the enums above
namespace Lookup {
// Define a template for a name table that maps enum values to string views
template <typename T, std::size_t N>
using NameTable = std::array<std::pair<T, std::string_view>, N>;

// Text from enum value
template <typename T, std::size_t N>
constexpr std::string_view name_of(const NameTable<T, N>& table, T value) {
    const auto entry = std::ranges::find(table, value, &std::pair<T, std::string_view>::first);
    return entry != table.end() ? entry->second : "Unknown";
}

// Enum value from text
template <typename T, std::size_t N>
constexpr std::optional<T> value_of(const NameTable<T, N>& table, std::string_view name) {
    const auto entry = std::ranges::find(table, name, &std::pair<T, std::string_view>::second);
    return entry != table.end() ? std::optional<T>(entry->first) : std::nullopt;
}

inline constexpr NameTable<ActivationType, 5> activations{{
    {ActivationType::RELU, "relu"},
    {ActivationType::SIGMOID, "sigmoid"},
    {ActivationType::TANH, "tanh"},
    {ActivationType::LINEAR, "linear"},
    {ActivationType::SOFTMAX, "softmax"},
}};

inline constexpr NameTable<OptimizerType, 3> optimizers{{
    {OptimizerType::SGD, "sgd"},
    {OptimizerType::ADAM, "adam"},
    {OptimizerType::RMSPROP, "rmsprop"},
}};

inline constexpr NameTable<LossType, 3> losses{{
    {LossType::MSE, "mse"},
    {LossType::BCE, "bce"},
    {LossType::CCE, "cce"},
}};

inline constexpr NameTable<InitType, 4> inits{{
    {InitType::RANDOM, "random"},
    {InitType::LECUN, "lecun"},
    {InitType::GLOROT, "glorot"},
    {InitType::HE, "he"},
}};

inline constexpr NameTable<TaskType, 2> tasks{{
    {TaskType::REGRESSION, "Regression"},
    {TaskType::CLASSIFICATION, "Classification"},
}};

inline constexpr NameTable<DatasetType, 10> datasets{{
    {DatasetType::XOR, "XOR"},
    {DatasetType::XOR_HOT, "XOR_HOT"},
    {DatasetType::MONK1, "MONK1"},
    {DatasetType::MONK1_HOT, "MONK1_HOT"},
    {DatasetType::MONK2, "MONK2"},
    {DatasetType::MONK2_HOT, "MONK2_HOT"},
    {DatasetType::MONK3, "MONK3"},
    {DatasetType::MONK3_HOT, "MONK3_HOT"},
    {DatasetType::MLCUP, "MLCUP"},
    {DatasetType::MNIST, "MNIST"},
}};

inline constexpr NameTable<StoppingRule, 2> stopping_rules{{
    {StoppingRule::CONVERGENCE, "convergence"},
    {StoppingRule::ERROR_LEVEL, "error"},
}};

inline constexpr NameTable<NormalizationType, 4> normalizations{{
    {NormalizationType::NONE, "none"},
    {NormalizationType::MIN_MAX, "minmax"},
    {NormalizationType::ABS_MAX, "max"},
    {NormalizationType::Z_SCORE, "zscore"},
}};

inline constexpr NameTable<ActivationType, 5> activation_labels{{
    {ActivationType::RELU, "ReLU"},
    {ActivationType::SIGMOID, "Sigmoid"},
    {ActivationType::TANH, "Tanh"},
    {ActivationType::LINEAR, "Linear"},
    {ActivationType::SOFTMAX, "Softmax"},
}};

inline constexpr NameTable<OptimizerType, 3> optimizer_labels{{
    {OptimizerType::SGD, "SGD"},
    {OptimizerType::ADAM, "Adam"},
    {OptimizerType::RMSPROP, "RMSProp"},
}};

inline constexpr NameTable<LossType, 3> loss_labels{{
    {LossType::MSE, "Mean Squared Error"},
    {LossType::BCE, "Binary Cross-Entropy"},
    {LossType::CCE, "Categorical Cross-Entropy"},
}};

inline constexpr NameTable<InitType, 4> init_labels{{
    {InitType::RANDOM, "Random"},
    {InitType::LECUN, "LeCun"},
    {InitType::GLOROT, "Glorot"},
    {InitType::HE, "He"},
}};

inline const std::map<std::string, DatasetType> str_to_dataset{
    {"xor", DatasetType::XOR},
    {"xor_hot", DatasetType::XOR_HOT},
    {"monk1", DatasetType::MONK1},
    {"monk1_hot", DatasetType::MONK1_HOT},
    {"monk2", DatasetType::MONK2},
    {"monk2_hot", DatasetType::MONK2_HOT},
    {"monk3", DatasetType::MONK3},
    {"monk3_hot", DatasetType::MONK3_HOT},
    {"mlcup", DatasetType::MLCUP},
    {"mnist", DatasetType::MNIST},
};

inline const std::map<std::string, StoppingRule> str_to_stopping{
    {"convergence", StoppingRule::CONVERGENCE},
    {"level", StoppingRule::ERROR_LEVEL},
};

inline const std::map<std::string, NormalizationType> str_to_normalization{
    {"none", NormalizationType::NONE},
    {"minmax", NormalizationType::MIN_MAX},
    {"max", NormalizationType::ABS_MAX},
    {"zscore", NormalizationType::Z_SCORE},
};
} // namespace Lookup
