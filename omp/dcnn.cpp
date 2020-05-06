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
  if (dest->n != res->n || dest->m != res->m){
    printf("linearForward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);
  return;
}

void linearBackward1(matrix_t dest, matrix_t a, matrix_t b) {
  matrix_t transA = transpose(a);
  matrix_t res = dot(b, transA);
  if (dest->n != res->n || dest->m != res->m){
    printf("linearBackward1 Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);
  matrix_free(transA);

  return;
}

void linearBackward2(matrix_t dest, matrix_t a, matrix_t b) {
  matrix_t trA = transpose(a);
  matrix_t slicedA = slice(trA, 1, trA->n);
  matrix_t res = dot(slicedA, b);
  if (dest->n != res->n || dest->m != res->m){
    printf("linearBackward2 Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);
  matrix_free(trA);
  matrix_free(slicedA);

  return;
}

void sigmForward(matrix_t dest, matrix_t v) {
  matrix_t res;
  size_t n = v->n;
  // Assert ns >= 1
  size_t m = v->m;

  res = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      res->data[i][j] = 1 / (1 + std::exp(-1 * v->data[i][j]));
    }
  }


  if (dest->n != res->n + 1 || dest->m != res->m){
    printf("sigmForward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row+1][col] = res->data[row][col];
    }
  }
  dest->data[0][0] = 1.0;
  matrix_free(res);

  return;
}

void sigmBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                             matrix_t gradActivation) {
  size_t n = gradActivation->n;
  size_t m = gradActivation->m;
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp->data[i + 1][j];
      res->data[i][j] = gradActivation->data[i][j] * temp * (1.0 - temp);
    }
  }

  if (dest->n != res->n || dest->m != res->m){
    printf("sigmBackward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);

  return;
}

void tanhForward(matrix_t dest, matrix_t v)
{
  matrix_t res = tanh(v);

  if (dest->n != res->n+1 || dest->m != res->m){
    printf("tanhForward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row+1][col] = res->data[row][col];
    }
  }
  dest->data[0][0] = 1.0;
  matrix_free(res);

  return;
}

void tanhBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                             matrix_t gradActivation)
{
  size_t n = gradActivation->n;
  size_t m = gradActivation->m;
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp->data[i + 1][j];
      res->data[i][j] = gradActivation->data[i][j] * (1.0 - temp*temp);
    }
  }

  if (dest->n != res->n || dest->m != res->m){
    printf("tanhBackward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);

  return;
}

void reluForward(matrix_t dest, matrix_t A)
{
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;

  matrix_t res = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      res->data[i][j] = std::max(0.0, A->data[i][j]);
    }
  }


  if (dest->n != res->n+1 || dest->m != res->m){
    printf("reluForward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row+1][col] = res->data[row][col];
    }
  }
  dest->data[0][0] = 1.0;
  matrix_free(res);

  return;
}

void reluBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                  matrix_t gradActivation)
{
  size_t n = gradActivation->n;
  size_t m = gradActivation->m;
  matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp->data[i + 1][j];
      res->data[i][j] = temp == 0.0 ? 0 : gradActivation->data[i][j];
    }
  }

  if (dest->n != res->n || dest->m != res->m){
    printf("reluBackward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);

  return;
}

void softForward(matrix_t dest, matrix_t v) {
  matrix_t exp_prev = exp(v);
  matrix_t res = divide(exp_prev, sum(exp_prev));


  if (dest->n != res->n || dest->m != res->m){
    printf("softForward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);
  matrix_free(exp_prev);

  return;
}

void softBackward(matrix_t dest, matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  matrix_t res = subtract(activationComp, y);

  if (dest->n != res->n || dest->m != res->m){
    printf("softBackward Wrong sizes %d %d %d %d\n", dest->n, dest->m, res->n, res->m);
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      dest->data[row][col] = res->data[row][col];
    }
  }
  matrix_free(res);

  return;
}

double crossEntropyForward(matrix_t v, matrix_t vh) {
  matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (size_t i = 0; i < v->n; i++) {
    crossEntropy -= v->data[i][0] * lvh->data[i][0];
  }
  return crossEntropy;
}

/* KERNALS HERE */

void single_epoch(matrix_t* X, matrix_t* Y, int num_samples,
  matrix_t* weights, layer_type_t* layer_types, int num_layers,
  double learning_rate, matrix_t** linearComp, matrix_t** activationComp,
  matrix_t** gradLinear, matrix_t** gradActivation, matrix_t** gradWeights) {

  int s, i;

  for (s = 0; s < num_samples; s += BATCH_SIZE) {
    #pragma omp parallel 
    {
      // #pragma omp for schedule(static)
      int b = omp_get_thread_num();
      // for (b = 0; b < BATCH_SIZE; b++)
      if (b < BATCH_SIZE) {
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
    }// end pragma omp parallel

    // update all the weights
    // #pragma omp parallel
    // {
      // #pragma omp for schedule(static)
      for (i = 0; i < num_layers; i++) {
        for (size_t b = 0; b < BATCH_SIZE; b++) {
          matrix_t scaMult = multiply(gradWeights[b][i], learning_rate);
          matrix_t diff = subtract(weights[i], scaMult);
          // update the weights vector manually
          for (size_t row = 0; row < diff->n; row++) {
            for (size_t col = 0; col < diff->m; col++) {
              weights[i]->data[row][col] = diff->data[row][col];
            }
          }
          matrix_free(scaMult);
          matrix_free(diff);
        }
      }
    // } // end pragma omp parallel
  }

  //weights_ = weights;
  //return weights_;

}




matrix_t* train(matrix_t* X, matrix_t* Y, int num_samples,
           int input_rows, int input_cols, int output_rows, int output_cols,
           int* num_units, layer_type_t* layer_types, int num_layers,
           double learning_rate, int num_epochs)
{
  // iterators
  int e, s, i, b;

  matrix_t* weights = (matrix_t *)calloc(sizeof(matrix_t), num_layers);

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

    weights[i] = randu(rows, cols);
  }

  // temp calculations
  matrix_t** linearComp = (matrix_t**)calloc(sizeof(matrix_t*), BATCH_SIZE);
  matrix_t** activationComp = (matrix_t**)calloc(sizeof(matrix_t*), BATCH_SIZE);
  matrix_t** gradLinear = (matrix_t**)calloc(sizeof(matrix_t*), BATCH_SIZE);
  matrix_t** gradActivation = (matrix_t**)calloc(sizeof(matrix_t*), BATCH_SIZE);
  matrix_t** gradWeights = (matrix_t**)calloc(sizeof(matrix_t*), BATCH_SIZE);
  for (b = 0; b < BATCH_SIZE; b++) {
    linearComp[b] = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
    activationComp[b] = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
    gradLinear[b] = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
    gradActivation[b] = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
    gradWeights[b] = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
    for (i = 0; i < num_layers; i++) {
      linearComp[b][i] = init(num_units[i], 1, 0.0);
      gradLinear[b][i] = init(num_units[i], 1, 0.0);
      if (i == 0){
        activationComp[b][i] = init(num_units[i] + 1, 1, 0.0);
        gradActivation[b][i] = init(num_units[i], 1, 0.0);
        gradWeights[b][i] = init(num_units[i], input_rows, 0.0);
      } else if (i == num_layers - 1) {
        activationComp[b][i] = init(num_units[i], 1, 0.0);
        gradActivation[b][i] = init(num_units[i], 1, 0.0);
        gradWeights[b][i] = init(output_rows, num_units[i - 1] + 1, 0.0);
      } else {
        activationComp[b][i] = init(num_units[i] + 1, 1, 0.0);
        gradActivation[b][i] = init(num_units[i], 1, 0.0);
        gradWeights[b][i] = init(num_units[i], num_units[i - 1] + 1, 0.0);
      }
    }
  }


  for (e = 0; e < num_epochs; e++) {
    single_epoch(X, Y, num_samples, weights, layer_types, num_layers, learning_rate,
      linearComp, activationComp, gradLinear, gradActivation, gradWeights);

    printf("epoch = %d avgEntropy = %f\n", e, 0.0);
  }

  for (b = 0; b < BATCH_SIZE; b++) {
    for (i = 0; i < num_layers; i++) {
      matrix_free(linearComp[b][i]);
      matrix_free(activationComp[b][i]);
      matrix_free(gradLinear[b][i]);
      matrix_free(gradActivation[b][i]);
      matrix_free(gradWeights[b][i]);
    }
    free(linearComp[b]);
    free(activationComp[b]);
    free(gradLinear[b]);
    free(gradActivation[b]);
    free(gradWeights[b]);
  }

  free(linearComp);
  free(activationComp);
  free(gradLinear);
  free(gradActivation);
  free(gradWeights);


  return weights;

}


size_t predict(matrix_t* weights, matrix_t x,
               int num_layers, layer_type_t* layer_types) {
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
