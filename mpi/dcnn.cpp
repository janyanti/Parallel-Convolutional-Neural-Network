/*
  dcnn.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include "dcnn.h"

using namespace matrix;


namespace dcnn {

/* INLINE DEVICE FUNCTIONS */

matrix_t linearForward(matrix_t a, matrix_t b) {
  return dot(b, a);
}

matrix_t linearBackward1(matrix_t a, matrix_t b) {
  return dot(b, transpose(a));
}

matrix_t linearBackward2(matrix_t a, matrix_t b) {
  matrix_t trA = transpose(a);
  matrix_t slicedA = slice(trA, 1, trA.size());
  return dot(slicedA, b);
}

matrix_t sigmForward(matrix_t v) {
  vec_t ones = init(1, 1.);
  matrix_t res;
  res = multiply(v, -1.);
  size_t n = v.size();
  size_t m = v[0].size();

  res = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      res[i][j] = 1 / (1 + std::exp(-1 * v[i][j]));
    }
  }
  res.insert(res.begin(), ones);
  return res;
}

matrix_t sigmBackward(matrix_t linearComp, matrix_t activationComp,
                             matrix_t gradActivation) {
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = gradActivation[i][j] * temp * (1.0 - temp);
    }
  }
  return res;
}

matrix_t tanhForward(matrix_t v)
{
  vec_t ones = init(1, 1.);
  matrix_t res = tanh(v);
  res.insert(res.begin(), ones);
  return res;
}

matrix_t tanhBackward(matrix_t linearComp, matrix_t activationComp,
                             matrix_t gradActivation)
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = gradActivation[i][j] * (1.0 - temp*temp);
    }
  }
  return res;
}

matrix_t reluForward(matrix_t A)
{
  size_t ns = A.size();
  size_t ms = A[0].size();

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = std::max(0.0, A[i][j]);
    }
  }
  vec_t ones = init(1, 1.);
  B.insert(B.begin(), ones);
  return B;
}

matrix_t reluBackward(matrix_t linearComp, matrix_t activationComp,
                             matrix_t gradActivation)
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = temp == 0.0 ? 0 : gradActivation[i][j];
    }
  }
  return res;
}

matrix_t softForward(matrix_t v) {
  matrix_t exp_prev = exp(v);
  return divide(exp_prev, sum(exp_prev));
}

matrix_t softBackward(matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  return subtract(activationComp, y);
}

double crossEntropyForward(matrix_t v, matrix_t vh) {
  matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (size_t i = 0; i < v.size(); i++) {
    crossEntropy -= v[i][0] * lvh[i][0];
  }
  return crossEntropy;
}

/* KERNALS HERE */

double forwardStep(std::vector<matrix_t> X, std::vector<matrix_t> Y,
            std::vector<matrix_t> weights,
            std::vector<layer_type_t> layer_types,
            int num_layers, int e) {


  int s, i;
  int num_samples = X.size();


  double totalEntropy = 0.0;
  for (s = 0; s < num_samples; s++) {
    matrix_t comp;
    for (i = 0; i < num_layers; i++) {
      if (i == 0)
        comp = linearForward(X[s], weights[i]);
      else
        comp = linearForward(comp, weights[i]);

      switch (layer_types[i]) {
        case SIGM:
          comp = sigmForward(comp);
          break;
        case SOFT:
          comp = softForward(comp);
          break;
        case TANH:
          comp = tanhForward(comp);
          break;
        case RELU:
          comp = reluForward(comp);
          break;
        default:
          comp = init(1, 1, 0.0);
          break;
      }
    }
    totalEntropy += crossEntropyForward(Y[s], comp);
  }
  printf("epoch = %d avgEntropy = %f\n", e, totalEntropy/num_samples);

}


void single_epoch(std::vector<matrix_t> X, std::vector<matrix_t> Y,
  std::vector<matrix_t> &weights, std::vector<matrix_t> &deltaWeights,
  std::vector<layer_type_t> layer_types,
  int num_layers, double learning_rate) {


  int s, i, b;

  // temp calculations
  std::vector<matrix_t> linearComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> activationComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradLinear(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradActivation(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradWeights(num_layers, init(1, 1, 0.0));


  for (s = 0; s < X.size(); s += BATCH_SIZE) {
    for (b = 0; b < BATCH_SIZE; b++) {
      // forward computation
      for (i = 0; i < num_layers; i++) {
        // linear
        if (i == 0)
          linearComp[i] = linearForward(X[s+b], weights[i]);
        else
          linearComp[i] = linearForward(activationComp[i-1], weights[i]);

        // activation
        switch (layer_types[i]) {
          case SIGM:
            activationComp[i] = sigmForward(linearComp[i]);
            break;
          case SOFT:
            activationComp[i] = softForward(linearComp[i]);
            break;
          case TANH:
            activationComp[i] = tanhForward(linearComp[i]);
            break;
          case RELU:
            activationComp[i] = reluForward(linearComp[i]);
            break;
          default:
            activationComp[i] = init(1, 1, 0.0);
            break;
          }
        }

      // backward computation
      for (i = num_layers - 1; i >= 0; i--) {

        switch (layer_types[i]) {
          case SIGM:
            gradLinear[i] = sigmBackward(linearComp[i], activationComp[i],
                                   gradActivation[i]);
            break;
          case SOFT:
            gradLinear[i] = softBackward(Y[s+b], linearComp[i],
                                   activationComp[i], gradActivation[i]);
            break;
          case TANH:
            gradLinear[i] = tanhBackward(linearComp[i], activationComp[i],
                                   gradActivation[i]);
            break;
          case RELU:
            gradLinear[i] = reluBackward(linearComp[i], activationComp[i],
                                   gradActivation[i]);
            break;
          default:
            gradLinear[i] = init(1, 1, 0.0);
            break;
        }

        if (i == 0) {
          gradWeights[i] = linearBackward1(X[s+b], gradLinear[i]);
        } else {
          gradWeights[i] =
              linearBackward1(activationComp[i - 1], gradLinear[i]);
          gradActivation[i - 1] = linearBackward2(weights[i], gradLinear[i]);
        }
      }

      for (i = 0; i < num_layers; i++) {
        deltaWeights[i] = add(deltaWeights[i], gradWeights[i]);
      }
    }

    // update all the weights
    for (i = 0; i < num_layers; i++) {
      deltaWeights[i] = divide(deltaWeights[i], BATCH_SIZE);
      weights[i] =
        subtract(weights[i], multiply(deltaWeights[i], learning_rate));
      clear(deltaWeights[i]);
    }
  }

}




void train(std::vector<matrix_t> &X, std::vector<matrix_t> &Y, int num_samples,
           int input_rows, int input_cols, int output_rows, int output_cols,
           std::vector<int> num_units,
           std::vector<layer_type_t> layer_types,
           int num_layers, double learning_rate, int num_epochs,
           std::vector<matrix_t> &weights)
{

  std::vector<matrix_t> deltaWeights;
  // iterators
  int e, i;

  // alloc weights
  int rows, cols;
  for (i = 0; i < num_layers; i++) {
    rows = num_units[i];
    if (i == num_layers - 1) {
      rows = output_rows;
    } else {
      rows = num_units[i];
    }

    if (i == 0) {
      cols = input_rows;
    } else {
      cols = num_units[i - 1] + 1;
    }

    weights.push_back(randu(rows, cols));
    deltaWeights.push_back(init(rows, cols, 0.0));
  }


  for (e = 0; e < num_epochs; e++) {
    single_epoch(X, Y, weights, deltaWeights, layer_types, num_layers, learning_rate);
    forwardStep(X, Y, weights, layer_types, num_layers, e);
  }


}


size_t predict(std::vector<matrix_t> &weights, matrix_t &x,
               int num_layers, std::vector<layer_type_t> layer_types) {
  // forward computation
  int i;
  matrix_t comp;
  for (i = 0; i < num_layers; i++) {
    if (i == 0)
      comp = linearForward(x, weights[i]);
    else
      comp = linearForward(comp, weights[i]);

    switch (layer_types[i]) {
    case SIGM:
      comp = sigmForward(comp);
      break;
    case SOFT:
      comp = softForward(comp);
      break;
    case TANH:
      comp = tanhForward(comp);
      break;
    case RELU:
      comp = reluForward(comp);
      break;
    default:
      comp = init(1, 1, 0.0);
      break;
    }
  }
  // now that we have the final computation we can turn it into a one-hot vector
  double maxVal = max(comp);
  for (size_t i = 0; i < comp.size(); i++) {
    if (comp[i][0] == maxVal) {
      return i;
    }
  }

  // should never reach this line
  return 0;
}


}; // namespace dcnn
