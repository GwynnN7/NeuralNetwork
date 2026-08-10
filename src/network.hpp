#pragma once

#include "cli.hpp"
#include "layer.hpp"
#include "metrics.hpp"
#include "model.hpp"
#include "normalizer.hpp"
#include "types.hpp"

#include <memory>
#include <vector>

// Forward declarations
struct Dataset;
struct DataSplit;

// Context for a training run
struct TrainContext {
    int trial = 0;         // Current trial number
    int epochs = 0;        // Number of epochs to train for
    Scalar patience = 0.0; // Patience for early stopping
    Scalar warmup = 0.0;   // Warmup ratio for learning rate

    NormalizationType norm_type = NormalizationType::NONE; // Normalization method
    StoppingRule stopping = StoppingRule::PATIENCE;        // Early stopping rule requested for this run
    std::optional<Scalar> target_error = std::nullopt;     // Training error level to stop at, only set when the ERROR rule has one to aim for

    int model_id = 0;    // Grid-search id of the model being trained
    int outer_index = 0; // Outer cross-validation fold
    int inner_index = 0; // Inner cross-validation fold

    bool in_model_selection = false; // Selecting on a validation fold rather than final training
    bool logging = true;             // Write the per-epoch metrics CSV

    static TrainContext from_args(const Args& args) {
        return TrainContext{
            .epochs = args.epochs,
            .patience = args.patience,
            .warmup = args.warmup,
            .norm_type = args.normalization_type,
            .stopping = args.stopping_rule};
    }
};

class Network {
  private:
    // The layers of the network (DenseLayer and ActivationLayer)
    std::vector<std::unique_ptr<Layer>> layers;
    LossPair loss_pair;

    // Normalizers fitted to the training data for features and labels
    Normalizer features_norm, labels_norm;

    void setLossFunction(LossType lossType);
    void addLayer(std::unique_ptr<Layer> layer);
    void buildLayers(const Model& model, int num_features, int num_classes, const std::vector<Parameters>* params, bool instantiate_optimizer);
    static void validateModel(const Model& model, int num_features, int num_classes);

    void snapshotParameters();
    void restoreParameters();

    Matrix forward(const Matrix& input);
    void backward(const Matrix& output_gradient, Scalar batch_fraction);

  public:
    explicit Network(const Model& model);
    Network(const Model& model, int num_features, int num_classes);
    Network(const Model& model, const std::vector<Parameters>& params, bool instantiate_optimizer = false);
    ~Network() = default;

    Model model;
    RunCurves train(const Dataset& dataset, const DataSplit& indices, const TrainContext& ctx);
    OUT Matrix predict(const Matrix& input) const;

    void setNormalizers(Normalizer input, Normalizer target);
    OUT const Normalizer& featuresNormalizer() const noexcept { return features_norm; }
    OUT const Normalizer& labelsNormalizer() const noexcept { return labels_norm; }

    OUT std::vector<const DenseLayer*> getDenseLayers() const;
};