#pragma once

#include "selection.hpp"
#include "summary.hpp"
#include "types.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct Model {
    int id = 0;
    std::vector<int> net_struct = {}; // Number of neurons in each hidden layer

    ActivationType hidden_activation = ActivationType::RELU;
    ActivationType output_activation = ActivationType::LINEAR;
    InitType init_type = InitType::GLOROT;
    OptimizerType opt_type = OptimizerType::SGD;
    LossType loss_type = LossType::MSE;

    int batch_size = 0;
    Scalar eta = 0.0;       // Learning rate
    Scalar lambda = 0.0;    // L2 regularization
    Scalar alpha = 0.0;     // SGD/RMSProp momentum
    Scalar beta1 = ADAM_B1; // Adam "momentum"

    // Runtime parameters, not part of model selection
    TaskType task = TaskType::REGRESSION;
    SelectionScore score = {};                      // Score of the model for model selection
    SplitSummary summary = {};                      // Summary of the model's performance across all folds and trials
    std::optional<SplitSummary> final_summary = {}; // Summary of the model's performance when (re)trained on the entire training set of the outer fold (either won model selection or was the only model available)

    void print(int epochs = 0) const;
    OUT static std::vector<Model> load_grid_search(const std::string& filename);
};

// Forward declarations
struct Dataset;
class Network;

namespace Serializer {
void dump_model(const std::filesystem::path& file, const Model& model, const Network& network);
OUT std::expected<std::unique_ptr<Network>, std::string> load_model(const std::filesystem::path& file, const Dataset& dataset);
} // namespace Serializer