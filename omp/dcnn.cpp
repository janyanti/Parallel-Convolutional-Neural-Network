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

void linearForward(matrix_t dest, matrix_t a, matrix_t b) {
  //printf("A DIMS: (%d, %d) \n", a.size(), a[0].size());
  //printf("B DIMS: (%d, %d) \n", b.size(), b[0].size());

  matrix_t res = dot(b, a);
  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void linearBackward1(matrix_t dest, matrix_t a, matrix_t b) {
  matrix_t transA = transpose(a);
  matrix_t res = dot(b, transA);
  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void linearBackward2(matrix_t dest, matrix_t a, matrix_t b) {
  matrix_t trA = transpose(a);
  matrix_t slicedA = slice(trA, 1, trA.size());
  matrix_t res = dot(slicedA, b);
  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void sigmForward(matrix_t dest, matrix_t v) {
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
  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void sigmBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
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

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void tanhForward(matrix_t dest, matrix_t v)
{
  vec_t ones = init(1, 1.);
  matrix_t res = tanh(v);
  res.insert(res.begin(), ones);

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void tanhBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
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

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void reluForward(matrix_t dest, matrix_t A)
{
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  matrix_t res = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      res[i][j] = std::max(0.0, A[i][j]);
    }
  }
  vec_t ones = init(1, 1.);
  res.insert(res.begin(), ones);

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void reluBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp, 
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

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void softForward(matrix_t dest, matrix_t v) {
  matrix_t exp_prev = exp(v);
  matrix_t res = divide(exp_prev, sum(exp_prev));


  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
}

void softBackward(matrix_t dest, matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  matrix_t res = subtract(activationComp, y);

  if (dest.size() != res.size() || dest[0].size() != res[0].size()){
    printf("Wrong sizes %d %d %d %d\n", dest.size(), dest[0].size(), res.size(), res[0].size());
  }

  for (size_t row = 0; row < res.size(); row++) {
    for (size_t col = 0; col < res[row].size(); col++) {
      dest[row][col] = res[row][col];
    }
  }

  return;
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

void single_epoch(std::vector<matrix_t> &X, std::vector<matrix_t> &Y,
  std::vector<matrix_t> &weights, std::vector<layer_type_t> &layer_types,
  int num_layers, double learning_rate, std::vector<tensor_t> &linearComp,
  std::vector<tensor_t> &activationComp, std::vector<tensor_t> &gradLinear,
  std::vector<tensor_t> &gradActivation, std::vector<tensor_t> &gradWeights) {

  int s, i, b;
  
  for (s = 0; s < X.size(); s += BATCH_SIZE) {
    // #pragma omp parallel 
    // {
    //   #pragma omp for schedule(static)
      for (b = 0; b < BATCH_SIZE; b++) {
        // forward computation
        for (i = 0; i < num_layers; i++) {
          // linear
          if (i == 0)
            linearForward(linearComp[b][i], X[s+b], weights[i]);
          else
            linearForward(linearComp[b][i], activationComp[b][i-1], weights[i]);


          // activation
          switch (layer_types[i]) {
            case SIGM:
            sigmForward(activationComp[b][i], linearComp[b][i]);
            break;
          case SOFT:
            softForward(activationComp[b][i], linearComp[b][i]);
            break;
          case TANH:
            tanhForward(activationComp[b][i], linearComp[b][i]);
            break;
          case RELU:
            reluForward(activationComp[b][i], linearComp[b][i]);
            break;
          default:
            break;
          }
        }
        // backward computation
        for (i = num_layers - 1; i >= 0; i--) {

          switch (layer_types[i]) {
          case SIGM:
            sigmBackward(gradLinear[b][i], linearComp[b][i], activationComp[b][i],
                                       gradActivation[b][i]);
            break;
          case SOFT:
            softBackward(gradLinear[b][i], Y[s+b], linearComp[b][i],
                                       activationComp[b][i], gradActivation[b][i]);
            break;
          case TANH:
            tanhBackward(gradLinear[b][i], linearComp[b][i], activationComp[b][i],
                                       gradActivation[b][i]);
            break;
          case RELU:
            reluBackward(gradLinear[b][i], linearComp[b][i], activationComp[b][i],
                                       gradActivation[b][i]);
            break;
          default:
            break;
          }

          if (i == 0) {
            linearBackward1(gradWeights[b][i], X[s+b], gradLinear[b][i]);
          } else {
            linearBackward1(gradWeights[b][i], activationComp[b][i - 1], gradLinear[b][i]);
            linearBackward2(gradActivation[b][i - 1], weights[i], gradLinear[b][i]);
          }
        }
      }
    // }// end pragma omp parallel
    
    // update all the weights
    // #pragma omp parallel 
    // {
      // #pragma omp for schedule(static)
      for (i = 0; i < num_layers; i++) {
        for (b = 0; b < BATCH_SIZE; b++) {
          matrix_t scaMult = multiply(gradWeights[b][i], learning_rate);
          matrix_t diff = subtract(weights[i], scaMult);
          // update the weights vector manually
          for (size_t row = 0; row < diff.size(); row++) {
            for (size_t col = 0; col < diff[row].size(); col++) {
              weights[i][row][col] = diff[row][col];
            }
          }
              
        }
      }
    // } // end pragma omp parallel
  }

  //weights_ = weights;
  //return weights_;

}




void train(std::vector<matrix_t> &X, std::vector<matrix_t> &Y, int num_samples,
           int input_rows, int input_cols, int output_rows, int output_cols,
           std::vector<int> num_units,
           std::vector<layer_type_t> layer_types,
           int num_layers, double learning_rate, int num_epochs,
           std::vector<matrix_t> &weights)
{
  // iterators
  int e, s, i, b;

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
  }

  // temp calculations
  std::vector<tensor_t> linearComp;
  std::vector<tensor_t> activationComp;
  std::vector<tensor_t> gradLinear;
  std::vector<tensor_t> gradActivation;
  std::vector<tensor_t> gradWeights;
  for (b = 0; b < BATCH_SIZE; b++) {
    tensor_t temp1;
    tensor_t temp2;
    tensor_t temp3;
    tensor_t temp4;
    tensor_t temp5;
    for (i = 0; i < num_layers; i++) {
      temp1.push_back(init(num_units[i], 1, 0.0));
      temp3.push_back(init(num_units[i], 1, 0.0));
      if (i == 0){
        temp2.push_back(init(num_units[i] + 1, 1, 0.0));
        temp4.push_back(init(num_units[i], 1, 0.0));
        temp5.push_back(init(num_units[i], input_rows, 0.0));
      } else if (i == num_layers - 1) {
        temp2.push_back(init(num_units[i], 1, 0.0));
        temp4.push_back(init(num_units[i], 1, 0.0));
        temp5.push_back(init(output_rows, num_units[i - 1] + 1, 0.0));
      } else {
        temp2.push_back(init(num_units[i] + 1, 1, 0.0));
        temp4.push_back(init(num_units[i], 1, 0.0));
        temp5.push_back(init(num_units[i], num_units[i - 1] + 1, 0.0));
      }
    }
    linearComp.push_back(temp1);
    activationComp.push_back(temp2);
    gradLinear.push_back(temp3);
    gradActivation.push_back(temp4);
    gradWeights.push_back(temp5);
  }


  for (e = 0; e < num_epochs; e++) {
    single_epoch(X, Y, weights, layer_types, num_layers, learning_rate,
      linearComp, activationComp, gradLinear, gradActivation, gradWeights);

    printf("epoch = %d avgEntropy = %f\n", e, 0.0);
  }


}


size_t predict(std::vector<matrix_t> &weights, matrix_t &x,
               int num_layers, std::vector<layer_type_t> layer_types) {
  /*
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
  */

  // should never reach this line
  return 0;
}


}; // namespace dcnn
