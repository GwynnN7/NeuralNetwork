#pragma once

#include "dataset.hpp"
#include "types.hpp"

#include <string>
#include <vector>

struct Model {
    int id = 0;
    std::vector<int> net_struct;
    ActivationType hidden_activation;
    ActivationType output_activation;
    InitType init_type;
    OptimizerType opt_type;
    LossType loss_type;

    int batch_size = 0;
    Scalar eta = 0.0;
    Scalar lambda = 0.0;
    Scalar alpha = 0.0;

    // Runtime parameters, not part of model selection
    TaskType task = TaskType::REGRESSION;

    void print() const;
    static std::vector<Model> load_grid_search(const std::string& filename);
};

class Network; // Forward declaration to avoid circular dependency

namespace Serializer {
void dump_model(const std::string& file, const Model& model, Network* network);
std::unique_ptr<Network> load_model(const std::string& file, const Dataset& dataset);
} // namespace Serializer