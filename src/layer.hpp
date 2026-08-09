#pragma once

#include "optimizer.hpp"
#include "types.hpp"

#include <memory>

class Layer {
  protected:
    Matrix X; // Input of the last training forward pass

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Model& model, Scalar batch_fraction, bool is_first_layer) = 0;

    // Snapshot and restore the layer's parameters in early stopping
    virtual void snapshot() {}
    virtual void restore() {}
};

class DenseLayer final : public Layer {
  private:
    Matrix W;
    Vector b;
    std::unique_ptr<Optimizer> optimizer;

    // Parameters of the best epoch
    struct Snapshot {
        Matrix weights;
        Vector biases;
    };
    std::optional<Snapshot> best;

    // Set the optimizer of the network based on the specified OptimizerType
    void setOptimizer(OptimizerType optType) {
        switch (optType) {
        case OptimizerType::SGD:
            optimizer = std::make_unique<GradientDescent>(W.rows(), W.cols());
            break;
        case OptimizerType::RMSPROP:
            optimizer = std::make_unique<RMSProp>(W.rows(), W.cols());
            break;
        case OptimizerType::ADAM:
            optimizer = std::make_unique<Adam>(W.rows(), W.cols());
            break;
        default:
            throw std::invalid_argument("Unsupported optimizer type");
        }
    }

  public:
    DenseLayer(int input_size, int output_size, InitType init_type, OptimizerType opt_type);
    DenseLayer(Matrix weights, Vector biases, OptimizerType opt_type, bool instantiate_optimizer);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model, Scalar batch_fraction, bool is_first_layer) override;

    void snapshot() override { best = Snapshot{W, b}; }
    void restore() override {
        if (best) {
            W = best->weights;
            b = best->biases;
        }
    }

    const Matrix& getWeights() const noexcept { return W; }
    const Vector& getBiases() const noexcept { return b; }
};

class ActivationLayer final : public Layer {
  private:
    ActivationPair activation;

  public:
    explicit ActivationLayer(ActivationType activation_type, bool derivative_in_loss = false);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model, Scalar batch_fraction, bool is_first_layer) override;
};