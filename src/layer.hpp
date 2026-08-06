#pragma once

#include "model.hpp"
#include "optimizer.hpp"
#include "types.hpp"

#include <memory>

class Layer {
  protected:
    Matrix X; // Input of the last training forward pass

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Model& model, bool is_first_layer) = 0;

    // Snapshot and restore the layer's parameters in early stopping
    virtual void snapshot() {}
    virtual void restore() {}
};

class DenseLayer : public Layer {
  private:
    Matrix W;
    Vector b;
    std::unique_ptr<Optimizer> optimizer;

    // Variables for early stopping behavior
    Matrix best_W;
    Vector best_b;
    bool has_snapshot = false;

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
    Matrix backward(const Matrix& output_gradient, const Model& model, bool is_first_layer) override;

    void snapshot() override {
        best_W = W;
        best_b = b;
        has_snapshot = true;
    }
    void restore() override {
        if (has_snapshot) {
            W = best_W;
            b = best_b;
        }
    }

    const Matrix& getWeights() const { return W; }
    const Vector& getBiases() const { return b; }
};

class ActivationLayer : public Layer {
  private:
    ActivationFunction activation;
    ActivationFunction activation_derivative;

    // True when the derivative is either handled by the loss or would do nothing
    bool skip_derivative;

  public:
    explicit ActivationLayer(ActivationType activation_type, bool derivative_in_loss = false);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model, bool is_first_layer) override;
};