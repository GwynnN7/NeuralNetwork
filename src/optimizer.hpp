#pragma once

#include "model.hpp"
#include "types.hpp"

#include <cmath>

class Optimizer {
  protected:
    // Apply the update each subclass already computed and add weight decay regularization directly to the weights
    void optimize(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model, Scalar batch_fraction) {
        // Scale the regularization term by the batch size
        const Scalar effective_lambda = model.lambda * batch_fraction;
        if (model.lambda > 0) {
            W *= (Scalar(1) - effective_lambda);
        }
        W += dW;
        b += db;
    }

  public:
    virtual ~Optimizer() = default;
    virtual void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model, Scalar batch_fraction) = 0;
};

class GradientDescent : public Optimizer {
  private:
    // Store the previous weight and bias updates for momentum
    Matrix d_W;
    Vector d_b;

  public:
    GradientDescent(Eigen::Index rows, Eigen::Index cols) {
        d_W = Matrix::Zero(rows, cols);
        d_b = Vector::Zero(rows);
    }

    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model, Scalar batch_fraction) override {
        // Calculate the weight and bias updates using momentum and learning rate
        d_W = -model.eta * dW + model.alpha * d_W;
        d_b = -model.eta * db + model.alpha * d_b;

        optimize(W, b, d_W, d_b, model, batch_fraction);
    }
};

class RMSProp : public Optimizer {
  private:
    // Store the moving average of squared gradients for weights and biases
    Matrix v_W;
    Vector v_b;
    // Reused buffers to avoid allocating new memory on each update
    Matrix d_W;
    Vector d_b;

  public:
    RMSProp(Eigen::Index rows, Eigen::Index cols) {
        v_W = d_W = Matrix::Zero(rows, cols);
        v_b = d_b = Vector::Zero(rows);
    }

    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model, Scalar batch_fraction) override {
        // Update the moving average of squared gradients for weights and biases
        v_W = (model.alpha * v_W) + (Scalar(1) - model.alpha) * dW.cwiseSquare();
        v_b = (model.alpha * v_b) + (Scalar(1) - model.alpha) * db.cwiseSquare();

        // Calculate the weight and bias updates using RMSProp algorithm
        d_W = -model.eta * dW.array() / (v_W.array().sqrt() + EPSILON);
        d_b = -model.eta * db.array() / (v_b.array().sqrt() + EPSILON);

        optimize(W, b, d_W, d_b, model, batch_fraction);
    }
};

class Adam : public Optimizer {
  private:
    // Store the moving average of gradients and squared gradients for weights and biases
    Matrix v_W, m_W;
    Vector v_b, m_b;
    // Reused buffers to avoid allocating new memory on each update
    Matrix d_W;
    Vector d_b;

    int t; // Time step for bias correction
    static constexpr Scalar B2 = ADAM_B2;

  public:
    Adam(Eigen::Index rows, Eigen::Index cols) {
        v_W = m_W = d_W = Matrix::Zero(rows, cols);
        v_b = m_b = d_b = Vector::Zero(rows);
        t = 0;
    }

    // Calculate the weight and bias updates using Adam algorithm (combination of momentum and RMSProp)
    void update(Matrix& W, Vector& b, const Matrix& dW, const Vector& db, const Model& model, Scalar batch_fraction) override {
        const Scalar B1 = model.beta1;

        // Update the moving average of gradients for weights and biases
        m_W = (B1 * m_W) + (Scalar(1) - B1) * dW;
        m_b = (B1 * m_b) + (Scalar(1) - B1) * db;

        // Update the moving average of squared gradients for weights and biases
        v_W = (B2 * v_W) + (Scalar(1) - B2) * dW.cwiseSquare();
        v_b = (B2 * v_b) + (Scalar(1) - B2) * db.cwiseSquare();

        // Bias correction
        t++;
        const Scalar m_corr = Scalar(1) - std::pow(B1, t);
        const Scalar v_corr = Scalar(1) - std::pow(B2, t);

        // Calculate the weight and bias updates using Adam algorithm
        d_W = -model.eta * (m_W.array() / m_corr) / ((v_W.array() / v_corr).sqrt() + EPSILON);
        d_b = -model.eta * (m_b.array() / m_corr) / ((v_b.array() / v_corr).sqrt() + EPSILON);

        optimize(W, b, d_W, d_b, model, batch_fraction);
    }
};