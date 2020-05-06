/*
  dcnn.cu

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include "dcnn.h"
#include <cuda.h>

using namespace matrix;


namespace dcnn {

/* INLINE DEVICE FUNCTIONS */
__global__
void dot(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double* A, double* B, double* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_A || j >= m_B) return;

  for (size_t k = 0; k < m_A; k++) {
    size_t idxA = i * m_A + k;
    size_t idxB = k * m_B + j;
    size_t idxC = i * m_B + j;
    C[idxC] += A[idxA] * B[idxB];
  }
  return;
}

__global__
void tanh(size_t n, size_t m, double* A, double* B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;

  size_t idx = i * m + j;
  if (idx == 0) {
    B[idx] = 1.0;
  } else {
    B[idx] = ::tanh(A[idx]);
  }

  return;
}

/* INLINE DEVICE FUNCTIONS */

/* CUDA DEVICE FUNCTIONS */
void cudalinearForward(matrix_t dest, dev_matrix_t _dest, dev_matrix_t a, dev_matrix_t b) {
  int threadsPerBlock = TPB;
  int blocks = ((_dest->n * _dest->m) + TPB - 1) / TPB;

  dot<<<blocks,threadsPerBlock>>>(b->n, b->m, a->n, a->m, b->_data, a->_data, _dest->_data);
  cudaDeviceSynchronize();

  cudaMemcpy(dest->data, _dest->_data, sizeof(double) * dest->n * dest->m, cudaMemcpyDeviceToHost);

}

void cudalinearBackward1(matrix_t dest, dev_matrix_t _dest, dev_matrix_t a, dev_matrix_t b) {
  int threadsPerBlock = TPB;
  int blocks = ((_dest->n * _dest->m) + TPB - 1) / TPB;

  /* Swapping a->n with a->m for transpose */
  dot<<<blocks,threadsPerBlock>>>(b->n, b->m, a->m, a->n, b->_data, a->_data, _dest->_data);
  cudaDeviceSynchronize();

  cudaMemcpy(dest->data, _dest->_data, sizeof(double) * _dest->n * _dest->m, cudaMemcpyDeviceToHost);
}

void cudatanhForward(matrix_t dest, dev_matrix_t _dest,  dev_matrix_t v) {
  int threadsPerBlock = TPB;
  int blocks = ((_dest->n * _dest->m) + TPB - 1) / TPB;

  tanh<<<blocks,threadsPerBlock>>>(_dest->n, _dest->m, v->_data, _dest->_data);
  cudaDeviceSynchronize();

  cudaMemcpy(dest->data, _dest->_data, sizeof(double) * _dest->n * _dest->m, cudaMemcpyDeviceToHost);

  return;
}


/* CUDA DEVICE FUNCTIONS */

void linearForward(matrix_t dest, matrix_t a, matrix_t b) {
  int threadsPerBlock = TPB;
  int blocks = ((dest->n * dest->m) + TPB - 1) / TPB;

  double *device_a, *device_b, *device_dest;

  cudaMalloc((void **)&device_a, sizeof(double) * a->n * a->m);
  cudaMalloc((void **)&device_b, sizeof(double) * b->n * b->m);
  cudaMalloc((void **)&device_dest, sizeof(double) * dest->n * dest->m);


  cudaMemcpy(device_a, a->data, sizeof(double) * a->n * a->m, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b->data, sizeof(double) * b->n * b->m, cudaMemcpyHostToDevice);

  dot<<<blocks,threadsPerBlock>>>(b->n, b->m, a->n, a->m, device_b, device_a, device_dest);
  cudaDeviceSynchronize();

  cudaMemcpy(dest->data, device_dest, sizeof(double) * dest->n * dest->m, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_dest);

  matrix_t res = dot(b, a);

  size_t idx;

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      idx = row * res->m + col;
      printf("Device: %lf  Host: %lf \n", dest->data[idx], res->data[idx]);
      dest->data[idx] = res->data[idx];
    }
    //printf("Device: %lf  Host: %lf \n", dest->data[idx], res->data[idx]);
  }
  matrix_free(res);
  return;
}

void linearBackward1(matrix_t dest, matrix_t a, matrix_t b) {
  matrix_t transA = transpose(a);
  matrix_t res = dot(b, transA);

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
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

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
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
      size_t idx = i * m + j;
      res->data[idx] = 1 / (1 + std::exp(-1 * v->data[idx]));
    }
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx + res->m] = res->data[idx];
    }
  }
  dest->data[0] = 1.0;
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
      size_t idx = i * m + j;
      double temp = activationComp->data[idx + m];
      res->data[idx] = gradActivation->data[idx] * temp * (1.0 - temp);
    }
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
    }
  }
  matrix_free(res);

  return;
}

void tanhForward(matrix_t dest, matrix_t v)
{
  matrix_t res = tanh(v);

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx + res->m] = res->data[idx];
    }
  }
  dest->data[0] = 1.0;
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
      size_t idx = i * m + j;
      double temp = activationComp->data[idx + m];
      res->data[idx] = gradActivation->data[idx] * (1.0 - temp*temp);
    }
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
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
      size_t idx = i * ms + j;
      res->data[idx] = ::max(0.0, A->data[idx]);
    }
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx + res->m] = res->data[idx];
    }
  }
  dest->data[0] = 1.0;
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
      size_t idx = i * m + j;
      double temp = activationComp->data[idx + m];
      res->data[idx] = temp == 0.0 ? 0 : gradActivation->data[idx];
    }
  }

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
    }
  }
  matrix_free(res);

  return;
}

void softForward(matrix_t dest, matrix_t v) {
  matrix_t exp_prev = exp(v);
  matrix_t res = divide(exp_prev, sum(exp_prev));

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
    }
  }
  matrix_free(res);
  matrix_free(exp_prev);

  return;
}

void softBackward(matrix_t dest, matrix_t y, matrix_t linearComp,
                             matrix_t activationComp, matrix_t gradActivation) {
  matrix_t res = subtract(activationComp, y);

  for (size_t row = 0; row < res->n; row++) {
    for (size_t col = 0; col < res->m; col++) {
      size_t idx = row * res->m + col;
      dest->data[idx] = res->data[idx];
    }
  }
  matrix_free(res);

  return;
}

double crossEntropyForward(matrix_t v, matrix_t vh) {
  matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (size_t i = 0; i < v->n; i++) {
    size_t idx = i * v->m;
    crossEntropy -= v->data[idx] * lvh->data[idx];
  }
  return crossEntropy;
}

/* KERNALS HERE */

void forwardStep(matrix_t* X, matrix_t* Y, matrix_t* weights,
            layer_type_t* layer_types, int* num_units, int num_layers,
            int num_samples, int e)
{

  int s, i;
  matrix_t* linearComp = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
  matrix_t* activationComp = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
  dev_matrix_t* _linearComp = (dev_matrix_t*)calloc(sizeof(matrix_t), num_layers);
  dev_matrix_t* _activationComp = (dev_matrix_t*)calloc(sizeof(matrix_t), num_layers);

  for (i = 0; i < num_layers; i++) {
    linearComp[i] = init(num_units[i], 1, 0.0);
    if (i == 0){
      activationComp[i] = init(num_units[i] + 1, 1, 0.0);
    } else if (i == num_layers - 1) {
      activationComp[i] = init(num_units[i], 1, 0.0);
    } else {
      activationComp[i] = init(num_units[i] + 1, 1, 0.0);
    }
  }

  double totalEntropy = 0.0;
  for (s = 0; s < num_samples; s++) {
    // forward computation
    for (i = 0; i < num_layers; i++) {
      // linear
      if (i == 0){
        linearForward(linearComp[i], X[s], weights[i]);
      }else
        linearForward(linearComp[i], activationComp[i-1], weights[i]);

      // activation
      switch (layer_types[i]) {
        case SIGM:
        sigmForward(activationComp[i], linearComp[i]);
        break;
      case SOFT:
        softForward(activationComp[i], linearComp[i]);
        break;
      case TANH:
        tanhForward(activationComp[i], linearComp[i]);
        break;
      case RELU:
        reluForward(activationComp[i], linearComp[i]);
        break;
      default:
        break;
      }
    }
    totalEntropy += crossEntropyForward(Y[s], activationComp[num_layers-1]);
  }

  for (size_t j = 0; j < num_layers; j++) {
    matrix_free(linearComp[j]);
    matrix_free(activationComp[j]);
  }
  free(linearComp);
  free(activationComp);

  printf("epoch = %d avgEntropy = %f\n", e, totalEntropy/num_samples);

}

void single_epoch(matrix_t* X, dev_matrix_t *_X, matrix_t* Y,
  dev_matrix_t *_Y, int num_samples,
  matrix_t* weights, dev_matrix_t *_weights, layer_type_t* layer_types, int num_layers,
  double learning_rate, matrix_t* linearComp, dev_matrix_t* _linearComp,
  matrix_t* activationComp, dev_matrix_t* _activationComp,
  matrix_t* gradLinear, dev_matrix_t* _gradLinear,
  matrix_t* gradActivation, dev_matrix_t* _gradActivation,
  matrix_t* gradWeights, dev_matrix_t* _gradWeights) {

  int s, i;

  for (s = 0; s < num_samples; s++) {
      // forward computation
      for (i = 0; i < num_layers; i++) {
        // linear
        if (i == 0) {
          // Working
          //cudalinearForward(linearComp[i], _linearComp[i], _X[s], _weights[i]);
          linearForward(linearComp[i], X[s], weights[i]);

        } else {
          linearForward(linearComp[i], activationComp[i-1], weights[i]);
          //cudalinearForward(linearComp[i], _linearComp[i], _activationComp[i-1], _weights[i]);
        }
        // activation
        switch (layer_types[i]) {
          case SIGM:
          sigmForward(activationComp[i], linearComp[i]);
          break;
        case SOFT:
          softForward(activationComp[i], linearComp[i]);
          break;
        case TANH:
          // Working
          //cudatanhForward(activationComp[i], _activationComp[i], _linearComp[i]);
          tanhForward(activationComp[i], linearComp[i]);
          break;
        case RELU:
          reluForward(activationComp[i], linearComp[i]);
          break;
        default:
          break;
        }
      }
      // backward computation
      for (i = num_layers - 1; i >= 0; i--) {

        switch (layer_types[i]) {
        case SIGM:
          sigmBackward(gradLinear[i], linearComp[i], activationComp[i],
                                     gradActivation[i]);
          break;
        case SOFT:
          softBackward(gradLinear[i], Y[s], linearComp[i],
                                     activationComp[i], gradActivation[i]);
          break;
        case TANH:
          tanhBackward(gradLinear[i], linearComp[i], activationComp[i],
                                     gradActivation[i]);
          break;
        case RELU:
          reluBackward(gradLinear[i], linearComp[i], activationComp[i],
                                     gradActivation[i]);
          break;
        default:
          break;
        }

        if (i == 0) {
          linearBackward1(gradWeights[i], X[s], gradLinear[i]);
        } else {
          linearBackward1(gradWeights[i], activationComp[i - 1], gradLinear[i]);
          linearBackward2(gradActivation[i - 1], weights[i], gradLinear[i]);
        }
      }


      for (i = 0; i < num_layers; i++) {
        matrix_t scaMult = multiply(gradWeights[i], learning_rate);
        matrix_t diff = subtract(weights[i], scaMult);
        // update the weights vector manually
        for (size_t row = 0; row < diff->n; row++) {
          for (size_t col = 0; col < diff->m; col++) {
            size_t idx = row * diff->m + col;
            weights[i]->data[idx] = diff->data[idx];
          }
        }
        matrix_free(scaMult);
        matrix_free(diff);
      }
  }

}


matrix_t* train(matrix_t* X, matrix_t* Y, int num_samples,
           int input_rows, int input_cols, int output_rows, int output_cols,
           int* num_units, layer_type_t* layer_types, int num_layers,
           double learning_rate, int num_epochs)
{
  // iterators
  int e, i;

  matrix_t* weights = (matrix_t *)calloc(sizeof(matrix_t), num_layers);
  dev_matrix_t* _weights = (dev_matrix_t *)calloc(sizeof(dev_matrix_t), num_layers);

  dev_matrix_t *_X = (dev_matrix_t *)calloc(sizeof(dev_matrix_t), num_samples);
  dev_matrix_t *_Y = (dev_matrix_t *)calloc(sizeof(dev_matrix_t), num_samples);

  size_t xn = X[0]->n;
  size_t xm = X[0]->m;

  size_t yn = Y[0]->n;
  size_t ym = Y[0]->m;

  for (size_t j = 0; j < num_samples; j++) {
    _X[j] = cuda_init(xn, xm, 0.0);
    cudaMemcpy(_X[j]->_data, X[j]->data, sizeof(double) * xn * xm, cudaMemcpyHostToDevice);
    _Y[j] = cuda_init(yn, ym, 0.0);
    cudaMemcpy(_Y[j]->_data, Y[j]->data, sizeof(double) * yn * ym, cudaMemcpyHostToDevice);
  }

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
    _weights[i] = cuda_init(rows, cols, 0.0);
    cudaMemcpy(_weights[i]->_data, weights[i]->data, sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
  }

  display(weights[num_layers-1]);

  // temp calculations
  matrix_t* linearComp = (matrix_t*)calloc(sizeof(matrix_t), 1);
  matrix_t* activationComp = (matrix_t*)calloc(sizeof(matrix_t), 1);
  matrix_t* gradLinear = (matrix_t*)calloc(sizeof(matrix_t), 1);
  matrix_t* gradActivation = (matrix_t*)calloc(sizeof(matrix_t), 1);
  matrix_t* gradWeights = (matrix_t*)calloc(sizeof(matrix_t), 1);

  dev_matrix_t* _linearComp = (dev_matrix_t*)calloc(sizeof(dev_matrix_t), 1);
  dev_matrix_t* _activationComp = (dev_matrix_t*)calloc(sizeof(dev_matrix_t), 1);
  dev_matrix_t* _gradLinear = (dev_matrix_t*)calloc(sizeof(dev_matrix_t), 1);
  dev_matrix_t* _gradActivation = (dev_matrix_t*)calloc(sizeof(dev_matrix_t), 1);
  dev_matrix_t* _gradWeights = (dev_matrix_t*)calloc(sizeof(dev_matrix_t), 1);

  for (i = 0; i < num_layers; i++) {
    linearComp[i] = init(num_units[i], 1, 0.0);
    _linearComp[i] = cuda_init(num_units[i], 1, 0.0);

    gradLinear[i] = init(num_units[i], 1, 0.0);
    _gradLinear[i] = cuda_init(num_units[i], 1, 0.0);

    if (i == 0){
      activationComp[i] = init(num_units[i] + 1, 1, 0.0);
      _activationComp[i] = cuda_init(num_units[i] + 1, 1, 0.0);

      gradActivation[i] = init(num_units[i], 1, 0.0);
      _gradActivation[i] = cuda_init(num_units[i], 1, 0.0);

      gradWeights[i] = init(num_units[i], input_rows, 0.0);
      _gradWeights[i] = cuda_init(num_units[i], input_rows, 0.0);

    } else if (i == num_layers - 1) {
      activationComp[i] = init(num_units[i], 1, 0.0);
      _activationComp[i] = cuda_init(num_units[i], 1, 0.0);

      gradActivation[i] = init(num_units[i], 1, 0.0);
      _gradActivation[i] = cuda_init(num_units[i], 1, 0.0);

      gradWeights[i] = init(output_rows, num_units[i - 1] + 1, 0.0);
      _gradWeights[i] = cuda_init(output_rows, num_units[i - 1] + 1, 0.0);

    } else {
      activationComp[i] = init(num_units[i] + 1,1, 0.0);
      _activationComp[i] = cuda_init(num_units[i] + 1, 1, 0.0);

      gradActivation[i] = init(num_units[i], 1, 0.0);
      _gradActivation[i] = cuda_init(num_units[i], 1, 0.0);

      gradWeights[i] = init(num_units[i], num_units[i - 1] + 1, 0.0);
      _gradWeights[i] = cuda_init(num_units[i], num_units[i - 1] + 1, 0.0);

    }
  }


  for (e = 0; e < num_epochs; e++) {
    single_epoch(X, _X, Y, _Y, num_samples, weights, _weights, layer_types, num_layers, learning_rate,
      linearComp, _linearComp, activationComp, _activationComp,
      gradLinear, _gradLinear, gradActivation, _gradActivation,
      gradWeights, _gradWeights);

    forwardStep(X, Y, weights, layer_types, num_units, num_layers, num_samples, e);

  }

  for (i = 0; i < num_layers; i++) {
    matrix_free(linearComp[i]);
    matrix_free(activationComp[i]);
    matrix_free(gradLinear[i]);
    matrix_free(gradActivation[i]);
    matrix_free(gradWeights[i]);
  }
  free(linearComp);
  free(activationComp);
  free(gradLinear);
  free(gradActivation);
  free(gradWeights);

  display(weights[num_layers-1]);

  return weights;

}


size_t predict(matrix_t* weights, matrix_t x, int* num_units,
             int num_layers, layer_type_t* layer_types) {
   // forward computation
   int i;

   matrix_t* linearComp = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
   matrix_t* activationComp = (matrix_t*)calloc(sizeof(matrix_t), num_layers);
   for (i = 0; i < num_layers; i++) {
     linearComp[i] = init(num_units[i], 1, 0.0);
     if (i == 0){
       activationComp[i] = init(num_units[i] + 1, 1, 0.0);
     } else if (i == num_layers - 1) {
       activationComp[i] = init(num_units[i], 1, 0.0);
     } else {
       activationComp[i] = init(num_units[i] + 1, 1, 0.0);
     }
   }


   // forward computation
   for (i = 0; i < num_layers; i++) {
     // linear
     if (i == 0){
       linearForward(linearComp[i], x, weights[i]);
     }else
       linearForward(linearComp[i], activationComp[i-1], weights[i]);

     // activation
     switch (layer_types[i]) {
       case SIGM:
       sigmForward(activationComp[i], linearComp[i]);
       break;
     case SOFT:
       softForward(activationComp[i], linearComp[i]);
       break;
     case TANH:
       tanhForward(activationComp[i], linearComp[i]);
       break;
     case RELU:
       reluForward(activationComp[i], linearComp[i]);
       break;
     default:
       break;
     }
   }


   // now that we have the final computation we can turn it into a one-hot vector
   matrix_t comp = activationComp[num_layers-1];
   double maxVal = max(comp);
   for (size_t i = 0; i < comp->n; i++) {
     size_t idx = i * comp->m;
     if (comp->data[idx] == maxVal) {
       for (size_t j = 0; j < num_layers; j++) {
         matrix_free(linearComp[j]);
         matrix_free(activationComp[j]);
       }
       free(linearComp);
       free(activationComp);
       return i;
     }
   }

   for (size_t j = 0; j < num_layers; j++) {
     matrix_free(linearComp[j]);
     matrix_free(activationComp[j]);
   }
   free(linearComp);
   free(activationComp);

   // should never reach this line
   return 0;
}


}; // namespace dcnn
