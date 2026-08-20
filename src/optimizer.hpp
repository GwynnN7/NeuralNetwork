#pragma once

#include "model.hpp"
#include "types.hpp"

#include <cmath>

/*
`alpha` hyperparameter covers SGD's momentum, RMSProp's "second moment" or Adam's first moment, `beta` hyperparameter covers Adam's second moment

SGD with Momentum (α = alpha):
  Update rule with momentum to smooth oscillations in gradient direction by following the previous gradient direction
  dw_tu = -eta * dEp/dw_tu + α * dw_tu_old

RMSProp (β = alpha):
  Second moment: moving average of the squared gradient to smooth oscillations in gradient magnitude
  v = β * v + (1 - β) * (dEp/dw_tu)^2
  Update rule:
  dw_tu = -eta * dEp/dw_tu / (sqrt(v) + epsilon)

Adam (β1 = alpha, β2 = beta):
  First moment: moving average of the gradient to smooth oscillations in gradient direction and avoid overshooting
  m = β1 * m + (1 - β1) * (dEp/dw_tu)
  Second moment: moving average of the squared gradient to smooth oscillations in gradient magnitude
  v = β2 * v + (1 - β2) * (dEp/dw_tu)^2
  Update rule with bias correction at step t, to avoid cold start problem (initialization at zero that causes the first updates to be too small):
  dw_tu = -eta * (m / (1 - β1^t)) / (sqrt(v / (1 - β2^t)) + epsilon)

Every optimizer applies L2 regularization independently of the learning rate and scaled by the mini-batch fraction:
  w_tu = (1 - lambda * (mb / l)) * w_tu

For RMSProp and Adam, L2 regularization is Decoupled Weight Decay (AdamW, applied directly to the weights) (1) and the denominator is bounded by EPSILON (2).
So all weights are regularized equally regardless of their past gradients (1) and very small gradients don't cause massive updates or division by zero (2).
*/

class Optimizer {
  protected:
    // Apply the update each subclass computed and add weight decay regularization directly to the weights
    void optimize(Parameters& params, const Parameters& gradient, const Model& model, Scalar decay_fraction) {
        /*
        Lambda is independent of eta and alpha, and each mini-batch applies only its own fraction of it (mb / l)
        This allows for better models tuning and keeps the effective regularization consistent across batch sizes
        But for long runs, the gradient gradually drops to near zero, while the decay does not, so the weights might collapse to zero
        */
        const Scalar effective_lambda = model.lambda * decay_fraction;
        if (model.lambda > 0) {
            params.W *= (Scalar(1) - effective_lambda);
        }
        params += gradient;
    }

  public:
    virtual ~Optimizer() = default;
    virtual void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar decay_fraction) = 0;
};

class GradientDescent : public Optimizer {
  private:
    // Store the previous gradient for momentum
    Parameters d_params;

  public:
    GradientDescent(Eigen::Index rows, Eigen::Index cols) {
        d_params = Parameters(rows, cols);
    }

    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar decay_fraction) override {
        // Calculate the weight and bias gradient using momentum and learning rate
        // dw_tu = eta * dEp/dw_tu + alpha * dw_tu_old
        d_params.W = -model.eta * gradient.W + model.alpha * d_params.W;
        d_params.b = -model.eta * gradient.b + model.alpha * d_params.b;

        optimize(params, d_params, model, decay_fraction);
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

    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar decay_fraction) override {
        const Scalar B = model.alpha;

        // Moving average of the squared gradient, v = β * v + (1 - β) * (dEp/dw_tu)^2
        v_params.W = (B * v_params.W) + (Scalar(1) - B) * gradient.W.cwiseSquare();
        v_params.b = (B * v_params.b) + (Scalar(1) - B) * gradient.b.cwiseSquare();

        // Calculate the weight and bias gradient using RMSProp algorithm
        // dw = -eta * dEp/dw_tu / (sqrt(v) + epsilon)
        d_params.W = -model.eta * gradient.W.array() / (v_params.W.array().sqrt() + EPSILON);
        d_params.b = -model.eta * gradient.b.array() / (v_params.b.array().sqrt() + EPSILON);

        optimize(params, d_params, model, decay_fraction);
    }
};

class Adam : public Optimizer {
  private:
    // Store the moving average of gradients and squared gradients
    Parameters v_params, m_params;
    // Reused buffers to avoid allocating new memory on each update
    Parameters d_params;

    int t; // Time step for bias correction

  public:
    Adam(Eigen::Index rows, Eigen::Index cols) {
        v_params = Parameters(rows, cols);
        m_params = Parameters(rows, cols);
        d_params = Parameters(rows, cols);
        t = 0;
    }

    // Calculate the weight and bias updates using Adam algorithm
    void update(Parameters& params, const Parameters& gradient, const Model& model, Scalar decay_fraction) override {
        const Scalar B1 = model.alpha;
        const Scalar B2 = model.beta;

        // First moment, m = β1 * m + (1 - β1) * (dEp/dw_tu)
        m_params.W = (B1 * m_params.W) + (Scalar(1) - B1) * gradient.W;
        m_params.b = (B1 * m_params.b) + (Scalar(1) - B1) * gradient.b;

        // Second moment, v = β2 * v + (1 - β2) * (dEp/dw_tu)^2
        v_params.W = (B2 * v_params.W) + (Scalar(1) - B2) * gradient.W.cwiseSquare();
        v_params.b = (B2 * v_params.b) + (Scalar(1) - B2) * gradient.b.cwiseSquare();

        // Bias correction to avoid cold start problem in the first iterations (zero initialization)
        t++;
        const Scalar m_corr = Scalar(1) - std::pow(B1, t); // m / (1 - β1^t)
        const Scalar v_corr = Scalar(1) - std::pow(B2, t); // v / (1 - β2^t)

        // Calculate the weight and bias updates using Adam algorithm
        // dw = -eta * m / (sqrt(v) + epsilon)
        d_params.W = -model.eta * (m_params.W.array() / m_corr) / ((v_params.W.array() / v_corr).sqrt() + EPSILON);
        d_params.b = -model.eta * (m_params.b.array() / m_corr) / ((v_params.b.array() / v_corr).sqrt() + EPSILON);

        optimize(params, d_params, model, decay_fraction);
    }
};