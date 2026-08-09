#pragma once

#include "types.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct Model {
    int id = 0;
    std::vector<int> net_struct;

    ActivationType hidden_activation = ActivationType::RELU;
    ActivationType output_activation = ActivationType::LINEAR;
    InitType init_type = InitType::GLOROT;
    OptimizerType opt_type = OptimizerType::SGD;
    LossType loss_type = LossType::MSE;

    int batch_size = 0;
    Scalar eta = 0.0;
    Scalar lambda = 0.0;
    Scalar alpha = 0.0;     // SGD/RMSProp momentum
    Scalar beta1 = ADAM_B1; // Adam "momentum"

    // Runtime parameters, not part of model selection
    TaskType task = TaskType::REGRESSION;

    void print() const;
    OUT static std::vector<Model> load_grid_search(const std::string& filename);
};

// Forward declarations
struct Dataset;
class Network;

namespace Serializer {
void dump_model(const std::filesystem::path& file, const Model& model, const Network& network);
OUT std::expected<std::unique_ptr<Network>, std::string> load_model(const std::filesystem::path& file, const Dataset& dataset);
} // namespace Serializer