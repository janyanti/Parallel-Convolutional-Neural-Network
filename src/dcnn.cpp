/*
  dcnn.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include "dcnn.h"

using namespace matrix;

namespace dcnn {

void model_t::train(std::vector<sample_t> samples, int num_samples, int input_rows,
                  int input_cols, int output_rows, int output_cols) {
  // iterators
  int  e, s, b, i;
  // temp calculations
  std::vector<matrix_t> linearComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> activationComp(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradLinear(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradActivation(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> gradWeights(num_layers, init(1, 1, 0.0));
  std::vector<matrix_t> deltaWeights;

  // alloc weights
  int rows, cols;
  for (i = 0; i < num_layers; i++) {
    int rows = num_units[i];
    if (i == num_layers - 1) {
      rows = output_rows;
    } else {
      rows = num_units[i];
    }

    if (i == 0) {
      cols = input_rows + 1;
    } else {
      cols = num_units[i - 1] + 1;
    }

    weights.push_back(randu(rows, cols));
    deltaWeights.push_back(init(rows, cols, 0.0));
  }

  for (e = 0; e < num_epochs; e++) {
    for (s = 0; s < num_samples; s += BATCH_SIZE) {
        for (b = 0; b < BATCH_SIZE; b++) {
          sample_t sample = samples[s+b];
          // printf("s = %d\n", s);
          // display(sample.getX());
          // display(sample.getY());
          // forward computation
          for (i = 0; i < num_layers; i++) {
            if (i == 0)
              linearComp[i] = linearForward(sample.getX(), weights[i]);
            else
              linearComp[i] = linearForward(activationComp[i-1], weights[i]);

            // printf("a/b = \n");
            // display(linearComp[i]);
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
            // printf("z/yh = \n");
            // display(activationComp[i]);
          }
          // double J = crossEntropyForward(sample.getY(), activationComp[num_layers - 1]);
          // printf("%f\n", J);

          // backward computation
          // double gJ = 1.0;
          // gradActivation[num_layers - 1] = crossEntropyBackward(samples.getY(),
          // activationComp[num_layers-1], J, gJ);
          // gradActivation[num_layers - 1] = ...
          // printf("gyh/gz = \n");
          // display(gradActivation[num_layers - 1]);
          for (i = num_layers - 1; i >= 0; i--) {
            switch (layer_types[i]) {
              case SIGM:
                gradLinear[i] = sigmBackward(linearComp[i], activationComp[i],
                                       gradActivation[i]);
                break;
              case SOFT:
                gradLinear[i] = softBackward(sample.getY(), linearComp[i],
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
            // printf("gb/ga = \n");
            // display(gradLinear[i]);
            if (i == 0) {
              gradWeights[i] = linearBackward1(sample.getX(), gradLinear[i]);
              // printf("gB/gA = \n");
              // display(gradWeights[i]);
            } else {
              gradWeights[i] =
              linearBackward1(activationComp[i - 1], gradLinear[i]);
              // printf("gB/gA = \n");
              // display(gradWeights[i]);
              gradActivation[i - 1] = linearBackward2(weights[i], gradLinear[i]);
              // printf("gyh/gz = \n");
              // display(gradActivation[i-1]);
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

    double totalEntropy = 0.0;
    for (s = 0; s < num_samples; s++) {
      matrix_t comp;
      for (i = 0; i < num_layers; i++) {
        if (i == 0)
          comp = linearForward(samples[s].getX(), weights[i]);
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
      totalEntropy += crossEntropyForward(samples[s].getY(), comp);
    }
    printf("epoch = %d avgEntropy = %f\n", e, totalEntropy/num_samples);
  }

  for (i = 0; i < num_layers; i++) {
    display(weights[i]);
  }
}

size_t model_t::predict(matrix_t x) {
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
  matrix_t yh = init(comp.size(), 1, 0.0);
  for (int i = 0; i < comp.size(); i++) {
    if (comp[i][0] == maxVal) {
      return i;
    }
  }

  // should never reach this line
  return 0;
}

matrix_t model_t::linearForward(matrix_t a, matrix_t b) {
  // printf("In linearForward x/z then A/B \n");
  // display(a);
  // display(b);
  return dot(b, a);
}

matrix_t model_t::linearBackward1(matrix_t a, matrix_t b) {
  return dot(b, transpose(a));
}

matrix_t model_t::linearBackward2(matrix_t a, matrix_t b) {
  matrix_t trA = transpose(a);
  matrix_t slicedA = slice(trA, 1, trA.size());
  return dot(slicedA, b);
}

matrix_t model_t::sigmForward(matrix_t v) {
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

matrix_t model_t::sigmBackward(matrix_t linearComp, matrix_t activationComp,
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


matrix_t model_t::tanhForward(matrix_t v)
{
  vec_t ones = init(1, 1.);
  matrix_t res = tanh(v);
  res.insert(res.begin(), ones);
  return res;
}

matrix_t model_t::tanhBackward(matrix_t linearComp, matrix_t activationComp,
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

matrix_t model_t::reluForward(matrix_t A)
{
  size_t ns = A.size();
  // Assert ns >= 1
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

matrix_t model_t::reluBackward(matrix_t linearComp, matrix_t activationComp,
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


matrix_t model_t::softForward(matrix_t v) {
  matrix_t exp_prev = exp(v);
  return divide(exp_prev, sum(exp_prev));
}

matrix_t model_t::softBackward(matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  return subtract(activationComp, y);
}

double model_t::crossEntropyForward(matrix_t v, matrix_t vh) {
  matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (int i = 0; i < v.size(); i++) {
    crossEntropy -= v[i][0] * lvh[i][0];
  }
  return crossEntropy;
}

// double model_t::crossEntropyBackward()
// {
//   return 0.0;
// }

}; // namespace dcnn
