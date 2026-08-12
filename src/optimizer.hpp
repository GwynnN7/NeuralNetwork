#pragma once

#include "model.hpp"
#include "types.hpp"

#include <cmath>

/*
SGD with Momentum:
  Update rule with momentum to smooth oscillations in gradient direction by following the previous gradient direction
  dw_tu = -eta * dEp/dw_tu + alpha * dw_tu_old

RMSProp:
  Second moment: moving average of the squared gradient to smooth oscillations in gradient magnitude
  v = alpha * v + (1 - alpha) * (dEp/dw_tu)^2
  Update rule:
  dw_tu = -eta * dEp/dw_tu / (sqrt(v) + epsilon)

Adam:
  First moment: moving average of the gradient to smooth oscillations in gradient direction and avoid overshooting
  m = beta1 * m + (1 - beta1) * (dEp/dw_tu)
  Second moment: moving average of the squared gradient to smooth oscillations in gradient magnitude
  v = beta2 * v + (1 - beta2) * (dEp/dw_tu)^2
  Update rule with bias correction at step t, to avoid cold start problem (initialization at zero that causes the first updates to be too small):
  dw_tu = -eta * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + epsilon)

Every optimizer applies L2 regularization independently of the learning rate and scaled by the mini-batch fraction:
  w_tu = (1 - lambda * (mb / l)) * w_tu

For RMSProp and Adam, L2 regularization is Decoupled Weight Decay (AdamW, applied directly to the weights) (1) and the denominator is bounded by EPSILON (2).
So all weights are regularized equally regardless of their past gradients (1) and very small gradients don't cause massive updates or division by zero (2).
*/

class Optimizer {
  protected:
    // Apply the update each subclass computed and add weight decay regularization directly to the weights
    void optimize(Parameters& params, const Parameters& gradient, const Model& model, Scalar batch_fraction) {
        // Lambda is independent of eta and alpha, and each mini-batch applies only its own fraction of it (mb / l)
        // This allows better models tuning and keeps the effective regularization consistent across batch sizes
        const Scalar effective_lambda = model.lambda * batch_fraction;
        if (model.lambda > 0) {
            params.W *= (Scalar(1) - effective_lambda);
        }
        params += gradient;
    }

  public:
    virtual ~Optimizer() = default;
    virtual void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar batch_fraction) = 0;
};

class GradientDescent : public Optimizer {
  private:
    // Store the previous gradient for momentum
    Parameters d_params;

  public:
    GradientDescent(Eigen::Index rows, Eigen::Index cols) {
        d_params = Parameters(rows, cols);
    }

    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar batch_fraction) override {
        // Calculate the weight and bias gradient using momentum and learning rate
        // dw_tu = eta * dEp/dw_tu + alpha * dw_tu_old
        d_params.W = -model.eta * gradient.W + model.alpha * d_params.W;
        d_params.b = -model.eta * gradient.b + model.alpha * d_params.b;

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

    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar batch_fraction) override {
        // Moving average of the squared gradient, v = alpha * v + (1 - alpha) * (dEp/dw_tu)^2
        v_params.W = (model.alpha * v_params.W) + (Scalar(1) - model.alpha) * gradient.W.cwiseSquare();
        v_params.b = (model.alpha * v_params.b) + (Scalar(1) - model.alpha) * gradient.b.cwiseSquare();

        // Calculate the weight and bias gradient using RMSProp algorithm
        // dw = -eta * dEp/dw_tu / (sqrt(v) + epsilon)
        d_params.W = -model.eta * gradient.W.array() / (v_params.W.array().sqrt() + EPSILON);
        d_params.b = -model.eta * gradient.b.array() / (v_params.b.array().sqrt() + EPSILON);

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
    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar batch_fraction) override {
        const Scalar B1 = model.beta1;

        // First moment, m = beta1 * m + (1 - beta1) * (dEp/dw_tu)
        m_params.W = (B1 * m_params.W) + (Scalar(1) - B1) * gradient.W;
        m_params.b = (B1 * m_params.b) + (Scalar(1) - B1) * gradient.b;

        // Second moment, v = beta2 * v + (1 - beta2) * (dEp/dw_tu)^2
        v_params.W = (B2 * v_params.W) + (Scalar(1) - B2) * gradient.W.cwiseSquare();
        v_params.b = (B2 * v_params.b) + (Scalar(1) - B2) * gradient.b.cwiseSquare();

        // Bias correction to avoid cold start problem in the first iterations (zero initialization)
        t++;
        const Scalar m_corr = Scalar(1) - std::pow(B1, t); // m / (1 - beta1^t)
        const Scalar v_corr = Scalar(1) - std::pow(B2, t); // v / (1 - beta2^t)

        // Calculate the weight and bias updates using Adam algorithm
        // dw = -eta * m / (sqrt(v) + epsilon)
        d_params.W = -model.eta * (m_params.W.array() / m_corr) / ((v_params.W.array() / v_corr).sqrt() + EPSILON);
        d_params.b = -model.eta * (m_params.b.array() / m_corr) / ((v_params.b.array() / v_corr).sqrt() + EPSILON);

        optimize(params, d_params, model, batch_fraction);
    }
};