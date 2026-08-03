#pragma once

#include "model.hpp"
#include "types.hpp"

class Optimizer {
  protected:
    // Update biases and weights (with L2 regularization)
    void optimize(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) {
        Matrix l2_penalty = model.eta * model.lambda * W;
        W += dW - l2_penalty;
        b += db;
    }

  public:
    virtual ~Optimizer() = default;
    virtual void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) = 0;
};

class GradientDescent : public Optimizer {
  private:
    // Store the previous weight and bias updates for momentum
    Matrix d_W;
    Vector d_b;

  public:
    GradientDescent(int rows, int cols) {
        d_W = Matrix::Zero(rows, cols);
        d_b = Vector::Zero(rows);
    }

    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) override {
        // Calculate the weight and bias updates using momentum and learning rate
        d_W = -model.eta * dW + model.alpha * d_W;
        d_b = -model.eta * db + model.alpha * d_b;

        optimize(W, b, d_W, d_b, model);
    }
};

class RMSProp : public Optimizer {
  private:
    // Store the moving average of squared gradients for weights and biases
    Matrix v_W;
    Vector v_b;

  public:
    RMSProp(int rows, int cols) {
        v_W = Matrix::Zero(rows, cols);
        v_b = Vector::Zero(rows);
    }

    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) override {
        // Update the moving average of squared gradients for weights and biases
        v_W = (model.alpha * v_W) + (1 - model.alpha) * dW.cwiseSquare();
        v_b = (model.alpha * v_b) + (1 - model.alpha) * db.cwiseSquare();

        // Calculate the weight and bias updates using RMSProp algorithm
        Matrix d_W = -model.eta * dW.array() / (v_W.array().sqrt() + EPSILON);
        Vector d_b = -model.eta * db.array() / (v_b.array().sqrt() + EPSILON);

        optimize(W, b, d_W, d_b, model);
    }
};

class Adam : public Optimizer {
  private:
    // Store the moving average of gradients and squared gradients for weights and biases
    Matrix v_W, m_W;
    Vector v_b, m_b;

    int t; // Time step for bias correction
    const Scalar B2 = ADAM_B2;

  public:
    Adam(int rows, int cols) {
        v_W = m_W = Matrix::Zero(rows, cols);
        v_b = m_b = Vector::Zero(rows);
        t = 0;
    }

    // Calculate the weight and bias updates using Adam algorithm (combination of momentum and RMSProp)
    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model) override {
        // Update the moving average of gradients for weights and biases
        m_W = (model.alpha * m_W) + (1.0 - model.alpha) * dW;
        m_b = (model.alpha * m_b) + (1.0 - model.alpha) * db;

        // Update the moving average of squared gradients for weights and biases
        v_W = (B2 * v_W) + (1 - B2) * dW.cwiseSquare();
        v_b = (B2 * v_b) + (1 - B2) * db.cwiseSquare();

        // Bias correction
        t++;
        Scalar m_corr = 1.0 - std::pow(model.alpha, t);
        Scalar v_corr = 1.0 - std::pow(B2, t);

        // Corrected moving averages
        Matrix m_W_corr = m_W / m_corr;
        Vector m_b_corr = m_b / m_corr;
        // Corrected moving averages of squared gradients
        Matrix v_W_corr = v_W / v_corr;
        Vector v_b_corr = v_b / v_corr;

        // Calculate the weight and bias updates using Adam algorithm
        Matrix d_W = -model.eta * m_W_corr.array() / (v_W_corr.array().sqrt() + EPSILON);
        Vector d_b = -model.eta * m_b_corr.array() / (v_b_corr.array().sqrt() + EPSILON);

        optimize(W, b, d_W, d_b, model);
    }
};