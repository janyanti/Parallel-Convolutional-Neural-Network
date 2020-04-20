/*
  dcnn.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include "dcnn.h"

using namespace matrix;

namespace dcnn {

void model::train(std::vector<sample_t> samples, int num_samples,
                  int input_rows, int input_cols, int output_rows,
                  int output_cols) {
  // iterators
  int e, s, i;
  // temp calculations
  std::vector<matrix_t> linearComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> activationComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradLinear(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradActivation(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradWeights(num_layers, init(1, 1, 0.0));

  // alloc weights
  std::vector<matrix_t> weights;
  int rows, cols;
  for (i = 0; i < num_layers; i++) {
    int rows = num_units[i];
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

    weights.push_back(init(rows, cols, 0.0));
  }

  for (e = 0; e < num_epochs; e++) {
    for (s = 0; s < num_samples; s++) {
      sample_t sample = samples[s];
      // forward computation
      for (i = 0; i < num_layers; i++) {
        linearComp[i] = linearForward(sample.x, weights[i]);
        switch (layer_types[i]) {
        case SIGM:
          activationComp[i] = sigmForward(linearComp[i]);
          break;
        case SOFT:
          activationComp[i] = softForward(linearComp[i]);
          break;
        // case TANH:
        //   activationComp[i] = tanhForward(linearComp[i]);
        //   break;
        // case RELU:
        //   activationComp[i] = reluForward(linearComp[i]);
        //   break;
        default:
          activationComp[i] = init(1, 1, 0.0);
          break;
        }
      }
      double J = crossEntropyForward(sample.y, activationComp[num_layers - 1]);

      // backward computation
      double gJ = 1.0;
      // gradActivation[num_layers - 1] = crossEntropyBackward(samples.y,
      // activationComp[num_layers-1], J, gJ);
      for (i = num_layers - 1; i > 0; i--) {
        switch (layer_types[i]) {
        case SIGM:
          gradLinear[i] =
              sigmBackward(linearComp[i], activationComp[i], gradActivation[i]);
          break;
        case SOFT:
          gradLinear[i] = softBackward(sample.y, linearComp[i],
                                       activationComp[i], gradActivation[i]);
          break;
        // case TANH:
        //   gradLinear[i] = tanhBackward(linearComp[i], activationComp[i],
        // case RELU:
        //   gradLinear[i] = reluBackward(linearComp[i], activationComp[i],
        default:
          gradLinear[i] = init(1, 1, 0.0);
          break;
        }
        if (i == 0)
          gradWeights[i] = linearBackward1(sample.x, gradLinear[i]);
        else
          gradWeights[i] =
              linearBackward1(activationComp[i - 1], gradLinear[i]);
        gradActivation[i - 1] = linearBackward2(weights[i], gradLinear[i]);
      }

      // update all the weights
      for (i = 0; i < num_layers; i++) {
        weights[i] =
            subtract(weights[i], multiply(gradWeights[i], learning_rate));
      }
    }
  }
}

matrix_t model::linearForward(matrix_t a, matrix_t b) { return dot(b, a); }

matrix_t model::linearBackward1(matrix_t a, matrix_t b) {
  return dot(b, transpose(a));
}

matrix_t model::linearBackward2(matrix_t a, matrix_t b) {
  matrix_t trA = transpose(a);
  return dot(slice(trA, 1, trA.size()), b);
}

matrix_t model::sigmForward(matrix_t v) {
  vec_t ones = init(1, 1.);
  matrix_t res;
  res = multiply(v, -1.);
  size_t n = v.size();
  // Assert ns >= 1
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

matrix_t model::sigmBackward(matrix_t linearComp, matrix_t activationComp,
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

// matrix_t model::tanhForward(matrix_t v)
// {
//   return;
// }
//
// matrix_t model::tanhBackward()
// {
//   return;
// }

// matrix_t model::reluForward(matrix_t v)
// {
//   return;
// }
//
// matrix_t model::reluBackward()
// {
//   return;
// }

matrix_t model::softForward(matrix_t v) {
  matrix_t exp_prev = exp(v);
  return divide(exp_prev, sum(exp_prev));
}

matrix_t model::softBackward(matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  return subtract(activationComp, y);
}

double model::crossEntropyForward(matrix_t v, matrix_t vh) { return 1.0; }

// double model::crossEntropyBackward()
// {
//   return;
// }

}; // namespace dcnn
