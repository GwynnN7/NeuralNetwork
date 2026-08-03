#pragma once

#include "dataset.hpp"
#include "metrics.hpp"
#include "model.hpp"
#include "optimizer.hpp"
#include "types.hpp"

class Layer {
  protected:
    Matrix X;
    Matrix Y;

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Model& model) = 0;
    virtual Scalar getWeightNorm() const { return 0.0; }
};

class DenseLayer : public Layer {
  private:
    Matrix W;
    Vector b;
    std::unique_ptr<Optimizer> optimizer;

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
    DenseLayer(Matrix weights, Vector biases, OptimizerType opt_type) : W(weights), b(biases) {
        setOptimizer(opt_type);
    }

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model) override;

    Matrix getWeights() const { return W; }
    Vector getBiases() const { return b; }
    Scalar getWeightNorm() const override { return W.squaredNorm(); }
};

class ActivationLayer : public Layer {
  private:
    ActivationFunction activation;
    ActivationFunction activation_derivative;

  public:
    ActivationLayer(ActivationType activationType);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, [[maybe_unused]] const Model& model) override;
};

class Network {
  private:
    std::vector<std::unique_ptr<Layer>> layers;
    LossFunction loss_func;
    LossDerivative loss_derivative;
    Scalar weights_norm;

    void setLossFunction(LossType lossType);
    void addLayer(std::unique_ptr<Layer> layer);
    void validateNetworkStructure(int num_classes) const;

  public:
    Network(const Model& model);
    Network(const Model& model, const int num_features, const int num_classes);
    Network(const Model& model, std::vector<Matrix> weights, std::vector<Vector> biases);
    ~Network() = default;

    Model model;

    Matrix predict(const Matrix& out, bool training = false);
    SplitResults train(const Dataset& dataset, const DataSplit& indices, int epochs, int model_index, int outer_index, int inner_index, bool logging = true);

    std::vector<const DenseLayer*> getDenseLayers() const;
    LossFunction getLossFunction() const { return loss_func; }
};