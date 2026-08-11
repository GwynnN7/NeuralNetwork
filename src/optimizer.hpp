#pragma once

#include "model.hpp"
#include "types.hpp"

#include <cmath>

class Optimizer {
  protected:
    // Apply the update each subclass computed and add weight decay regularization directly to the weights
    void optimize(Parameters& params, const Parameters& updates, const Model& model, Scalar batch_fraction) {
        // Lambda is independent of eta and alpha, and each mini-batch applies only its own fraction of it (mb / l) to keep the effective regularization consistent across batch sizes
        const Scalar effective_lambda = model.lambda * batch_fraction;
        if (model.lambda > 0) {
            params.W *= (Scalar(1) - effective_lambda);
        }
        params += updates;
    }

  public:
    virtual ~Optimizer() = default;
    virtual void update(Parameters& params, const Parameters& updates, const Model& model, Scalar batch_fraction) = 0;
};

class GradientDescent : public Optimizer {
  private:
    // Store the previous updates for momentum
    Parameters d_params;

  public:
    GradientDescent(Eigen::Index rows, Eigen::Index cols) {
        d_params = Parameters(rows, cols);
    }

    void update(Parameters& params, const Parameters& updates, const Model& model, Scalar batch_fraction) override {
        // Calculate the weight and bias updates using momentum and learning rate
        // Dw_tu = eta * delta_t * o_u + alpha * Dw_tu_old
        d_params.W = -model.eta * updates.W + model.alpha * d_params.W;
        d_params.b = -model.eta * updates.b + model.alpha * d_params.b;

        optimize(params, d_params, model, batch_fraction);
    }
};

class RMSProp : public Optimizer {
  private:
    // Store the moving average of squared gradients
    Parameters v_params;
    // Reused buffers to avoid allocating new memory on each update
    Parameters d_params;

  public:
    RMSProp(Eigen::Index rows, Eigen::Index cols) {
        v_params = Parameters(rows, cols);
        d_params = Parameters(rows, cols);
    }

    void update(Parameters& params, const Parameters& updates, const Model& model, Scalar batch_fraction) override {
        // Moving average of the squared gradient, v = alpha * v + (1 - alpha) * (dE/dw)^2
        v_params.W = (model.alpha * v_params.W) + (Scalar(1) - model.alpha) * updates.W.cwiseSquare();
        v_params.b = (model.alpha * v_params.b) + (Scalar(1) - model.alpha) * updates.b.cwiseSquare();

        // Calculate the weight and bias updates using RMSProp algorithm
        // Dw = -eta * (dE/dw) / (sqrt(v) + epsilon)
        d_params.W = -model.eta * updates.W.array() / (v_params.W.array().sqrt() + EPSILON);
        d_params.b = -model.eta * updates.b.array() / (v_params.b.array().sqrt() + EPSILON);

        optimize(params, d_params, model, batch_fraction);
    }
};

class Adam : public Optimizer {
  private:
    // Store the moving average of gradients and squared gradients
    Parameters v_params, m_params;
    // Reused buffers to avoid allocating new memory on each update
    Parameters d_params;

    int t; // Time step for bias correction
    static constexpr Scalar B2 = ADAM_B2;

  public:
    Adam(Eigen::Index rows, Eigen::Index cols) {
        v_params = Parameters(rows, cols);
        m_params = Parameters(rows, cols);
        d_params = Parameters(rows, cols);
        t = 0;
    }

    // Calculate the weight and bias updates using Adam algorithm (combination of momentum and RMSProp)
    void update(Parameters& params, const Parameters& updates, const Model& model, Scalar batch_fraction) override {
        const Scalar B1 = model.beta1;

        // First moment, m = beta1 * m + (1 - beta1) * (dE/dw)
        m_params.W = (B1 * m_params.W) + (Scalar(1) - B1) * updates.W;
        m_params.b = (B1 * m_params.b) + (Scalar(1) - B1) * updates.b;

        // Second moment, v = beta2 * v + (1 - beta2) * (dE/dw)^2
        v_params.W = (B2 * v_params.W) + (Scalar(1) - B2) * updates.W.cwiseSquare();
        v_params.b = (B2 * v_params.b) + (Scalar(1) - B2) * updates.b.cwiseSquare();

        // Bias correction to avoid cold start problem in the first iterations (zero initialization)
        t++;
        const Scalar m_corr = Scalar(1) - std::pow(B1, t);
        const Scalar v_corr = Scalar(1) - std::pow(B2, t);

        // Calculate the weight and bias updates using Adam algorithm
        // Dw = -eta * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + epsilon)
        d_params.W = -model.eta * (m_params.W.array() / m_corr) / ((v_params.W.array() / v_corr).sqrt() + EPSILON);
        d_params.b = -model.eta * (m_params.b.array() / m_corr) / ((v_params.b.array() / v_corr).sqrt() + EPSILON);

        optimize(params, d_params, model, batch_fraction);
    }
};