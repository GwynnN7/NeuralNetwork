#pragma once

#include "dataset.hpp"
#include "functions.hpp"
#include "types.hpp"

struct Model {
    int id = 0;
    std::vector<int> net_struct;
    ActivationType hidden_activation;
    ActivationType output_activation;
    InitType init_type;

    int epochs = 0;
    int batch_size = 0;
    Scalar eta = 0.0;
    Scalar lambda = 0.0;
    Scalar alpha = 0.0;
};

std::vector<Model> load_grid_search(const std::string& filename);

class Layer {
  protected:
    Matrix X;
    Matrix Y;

  public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input_matrix, bool training) = 0;
    virtual Matrix backward(const Matrix& output_gradient, const Model& model) = 0;
    virtual Scalar weightNorm() const { return 0.0; }
};

class DenseLayer : public Layer {
  private:
    Matrix W;
    Vector b;
    Matrix delta_W;

  public:
    DenseLayer(int input_size, int output_size, InitType init_type);
    DenseLayer(Matrix weights, Vector biases) : W(weights), b(biases) {
        delta_W = Matrix::Zero(W.rows(), W.cols());
    }

    Matrix forward(const Matrix& input_matrix, bool training) override;
    Matrix backward(const Matrix& output_gradient, const Model& model) override;

    Matrix getWeights() const { return W; }
    Vector getBiases() const { return b; }
    Scalar weightNorm() const override { return W.squaredNorm(); }
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