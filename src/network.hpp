#pragma once

#include "layer.hpp"
#include "metrics.hpp"
#include "model.hpp"
#include "types.hpp"

#include <memory>
#include <vector>

// Forward declarations
struct Dataset;
struct DataSplit;

// Context for a training run
struct TrainContext {
    int trial = 0;
    int epochs = 0;
    Scalar patience = 0.0; // Patience for early stopping
    Scalar warmup = 0.0;   // Warmup ratio for learning rate

    int model_id = 0;    // Grid-search id of the model being trained
    int outer_index = 0; // Outer cross-validation fold
    int inner_index = 0; // Inner cross-validation fold

    bool in_model_selection = false; // Selecting on a validation fold rather than final training
    bool logging = true;             // Write the per-epoch metrics CSV
};

class Network {
  private:
    std::vector<std::unique_ptr<Layer>> layers;
    LossPair loss_pair;

    void setLossFunction(LossType lossType);
    void addLayer(std::unique_ptr<Layer> layer);
    void buildLayers(const Model& model, int num_features, int num_classes, const std::vector<Matrix>* weights, const std::vector<Vector>* biases, bool build_optimizer);
    static void validateModel(const Model& model, int num_features, int num_classes);

    void snapshotParameters();
    void restoreParameters();

    Matrix forward(const Matrix& input);
    void backward(const Matrix& output_gradient, Scalar batch_fraction);

  public:
    explicit Network(const Model& model);
    Network(const Model& model, int num_features, int num_classes);
    Network(const Model& model, const std::vector<Matrix>& weights, const std::vector<Vector>& biases, bool build_optimizer = false);
    ~Network() = default;

    Model model;

    QUERY Matrix predict(const Matrix& input) const;
    RunCurves train(const Dataset& dataset, const DataSplit& indices, const TrainContext& ctx);

    QUERY std::vector<const DenseLayer*> getDenseLayers() const;
};