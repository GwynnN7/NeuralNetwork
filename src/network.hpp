#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <fstream>

class Layer {
  protected:
    Matrix X;
    Matrix Y;

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Args& args) = 0;
    virtual Scalar weight_norm() const { return 0.0; }
};

class DenseLayer : public Layer {
  private:
    Matrix W;
    Vector b;
    Matrix delta_W;

  public:
    DenseLayer(int input_size, int output_size, InitializationType init_type);
    DenseLayer(Matrix weights, Vector biases) : W(weights), b(biases) {
        delta_W = Matrix::Zero(W.rows(), W.cols());
    }

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Args& args) override;

    Matrix getWeights() const { return W; }
    Vector getBiases() const { return b; }
    Scalar weight_norm() const override { return W.squaredNorm(); }
};

class ActivationLayer : public Layer {
  private:
    std::function<Matrix(const Matrix&)> activation;
    std::function<Matrix(const Matrix&)> activation_derivative;

  public:
    ActivationLayer(ActivationType activationType);

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, [[maybe_unused]] const Args& args) override;
};

class Network {
  private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::function<Scalar(const Matrix&, const Matrix&)> loss_func;
    std::function<Matrix(const Matrix&, const Matrix&)> loss_derivative;
    Scalar weights_norm;

    Args args;
    std::ofstream log_file;

  public:
    Network(const Args& cli_args);
    Network(const Args& cli_args, const int num_features, const int num_classes);
    Network(const Args& cli_args, std::vector<Matrix> weights, std::vector<Vector> biases);
    ~Network();

    void setLossFunction(LossType lossType);
    void addLayer(Layer* layer);
    Matrix predict(Matrix out, bool training = false);
    void train(const ModelSet& model_set);

    std::vector<const DenseLayer*> getDenseLayers() const;
};