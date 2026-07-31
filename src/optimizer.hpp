#pragma once

#include "model.hpp"
#include "types.hpp"

class Optimizer {
  public:
    virtual ~Optimizer() = default;
    virtual void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) = 0;
};

class GradientDescent : public Optimizer {
  private:
    Matrix delta_W;

  public:
    GradientDescent(int rows, int cols) {
        delta_W = Matrix::Zero(rows, cols);
    }

    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) override {
        delta_W = -model.eta * dW + model.alpha * delta_W; // Update delta_W with learning rate and momentum
        Matrix l2_penalty = model.eta * model.lambda * W;  // L2 regularization term

        W = W + delta_W - l2_penalty; // Update weights with L2 regularization and momentum
        b -= model.eta * db;          // Update biases with learning rate
    }
};