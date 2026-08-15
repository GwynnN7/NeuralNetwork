#pragma once

#include "optimizer.hpp"
#include "types.hpp"

#include <memory>

/*
Layer:
  Layer classes are templates for each actual layer in the network
  Forward pass: computes the output of the layer given an input matrix
  Backward pass: computes the gradient of the loss given the output gradient

DenseLayer:
  A fully connected layer with weights and biases (Parameters struct)
  In backward pass updates the weights and biases using the optimizer

ActivationLayer:
  Follows each DenseLayer and applies a non-linear activation function to the input
*/

class Layer {
  protected:
    Matrix X; // Input of the last training forward pass

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Model& model, Scalar decay_fraction, bool is_first_layer) = 0;

    // Snapshot and restore the layer's parameters in early stopping
    virtual void takeSnapshot() {}
    virtual void restoreSnapshot() {}
};

class DenseLayer final : public Layer {
  private:
    // The weights and biases of the layer
    Parameters params;
    // The optimizer for this layer
    std::unique_ptr<Optimizer> optimizer;
    // Parameters of the best epoch
    std::optional<Parameters> snapshot;

    // Set the optimizer of the network based on the specified OptimizerType
    void setOptimizer(OptimizerType optType) {
        switch (optType) {
        case OptimizerType::SGD:
            optimizer = std::make_unique<GradientDescent>(params.W.rows(), params.W.cols());
            break;
        case OptimizerType::RMSPROP:
            optimizer = std::make_unique<RMSProp>(params.W.rows(), params.W.cols());
            break;
        case OptimizerType::ADAM:
            optimizer = std::make_unique<Adam>(params.W.rows(), params.W.cols());
            break;
        default:
            throw std::invalid_argument("Unsupported optimizer type");
        }
    }

  public:
    DenseLayer(int input_size, int output_size, InitType init_type, OptimizerType opt_type);
    DenseLayer(Parameters params, OptimizerType opt_type, bool instantiate_optimizer);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model, Scalar decay_fraction, bool is_first_layer) override;

    void takeSnapshot() override { snapshot = params; }
    void restoreSnapshot() override {
        // Restore the parameters from the snapshot if it exists
        if (snapshot) {
            params = *snapshot;
        }
    }

    const Parameters& getParameters() const noexcept { return params; }
};

class ActivationLayer final : public Layer {
  private:
    ActivationPair activation;

  public:
    explicit ActivationLayer(ActivationType activation_type, bool derivative_in_loss = false);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model, Scalar decay_fraction, bool is_first_layer) override;
};