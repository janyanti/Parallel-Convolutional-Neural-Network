#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/*
  dcnn.cu

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include "dcnn.h"

using namespace matrix;

/* INLINE DEVICE FUNCTIONS */

inline __device__
device_matrix_t linearForward(device_matrix_t a, device_matrix_t b) {
  return dot(b, a);
}

inline __device__
device_matrix_t linearBackward1(device_matrix_t a, device_matrix_t b) {
  return dot(b, transpose(a));
}

inline __device__
device_matrix_t linearBackward2(device_matrix_t a, device_matrix_t b) {
  device_matrix_t trA = transpose(a);
  device_matrix_t slicedA = slice(trA, 1, trA.size());
  return dot(slicedA, b);
}

inline __device__
device_matrix_t sigmForward(device_matrix_t v) {
  vec_t ones = device_init(1, 1.);
  device_matrix_t res;
  res = multiply(v, -1.);
  size_t n = v.size();
  // Assert ns >= 1
  size_t m = v[0].size();

  res = device_init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      res[i][j] = 1 / (1 + thrust::exp(-1 * v[i][j]));
    }
  }
  res.insert(res.begin(), ones);
  return res;
}

inline __device__
device_matrix_t sigmBackward(device_matrix_t linearComp, device_matrix_t activationComp,
                             device_matrix_t gradActivation) {
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  device_matrix_t res = device_init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = gradActivation[i][j] * temp * (1.0 - temp);
    }
  }
  return res;
}

inline __device__
device_matrix_t tanhForward(device_matrix_t v)
{
  vec_t ones = device_init(1, 1.);
  device_matrix_t res = tanh(v);
  res.insert(res.begin(), ones);
  return res;
}

inline __device__
device_matrix_t tanhBackward(device_matrix_t linearComp, device_matrix_t activationComp,
                             device_matrix_t gradActivation)
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  device_matrix_t res = device_init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = gradActivation[i][j] * (1.0 - temp*temp);
    }
  }
  return res;
}

inline __device__
device_matrix_t reluForward(device_matrix_t A)
{
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  device_matrix_t B = device_init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = thrust::maximum<double>(0.0, A[i][j]);
    }
  }
  vec_t ones = device_init(1, 1.);
  B.insert(B.begin(), ones);
  return B;
}

inline __device__
device_matrix_t reluBackward(device_matrix_t linearComp, device_matrix_t activationComp,
                             device_matrix_t gradActivation)
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  device_matrix_t res = device_init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = temp == 0.0 ? 0 : gradActivation[i][j];
    }
  }
  return res;
}

inline __device__
device_matrix_t softForward(device_matrix_t v) {
  device_matrix_t exp_prev = exp(v);
  return divide(exp_prev, sum(exp_prev));
}

inline __device__
device_matrix_t softBackward(device_matrix_t y, device_matrix_t linearComp,
                             device_matrix_t activationComp, device_matrix_t gradActivation) {
  return subtract(activationComp, y);
}

inline __device__
double crossEntropyForward(device_matrix_t v, device_matrix_t vh) {
  device_matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (int i = 0; i < v.size(); i++) {
    crossEntropy -= v[i][0] * lvh[i][0];
  }
  return crossEntropy;
}

/* KERNALS HERE */

__global__ void 
single_epoch_kernel(thrust::device_vector<device_matrix_t> device_weights, 
  thrust::device_vector<device_sample_t> device_samples, 
  int num_layers, double learning_rate) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;

  // temp calculations
  thrust::device_vector<device_matrix_t> linearComp(num_layers, device_init(1, 1, 0.0));
  thrust::device_vector<device_matrix_t> activationComp(num_layers, device_init(1, 1, 0.0));
  thrust::device_vector<device_matrix_t> gradLinear(num_layers, device_init(1, 1, 0.0));
  thrust::device_vector<device_matrix_t> gradActivation(num_layers, device_init(1, 1, 0.0));
  thrust::device_vector<device_matrix_t> gradWeights(num_layers, device_init(1, 1, 0.0));
  
  device_sample_t sample = device_samples[s];
  // forward computation
  for (i = 0; i < num_layers; i++) {
    // linear
    if (i == 0)
      linearComp[i] = linearForward(sample[0], device_weights[i]);
    else
      linearComp[i] = linearForward(activationComp[i-1], device_weights[i]);

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
      activationComp[i] = device_init(1, 1, 0.0);
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
      gradLinear[i] = softBackward(sample[1], linearComp[i],
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
      gradLinear[i] = device_init(1, 1, 0.0);
      break;
    }

    if (i == 0) {
      gradWeights[i] = linearBackward1(sample[0], gradLinear[i]);
    } else {
      gradWeights[i] =
          linearBackward1(activationComp[i - 1], gradLinear[i]);
      gradActivation[i - 1] = linearBackward2(weights[i], gradLinear[i]);
    }
  }

  // update all the device_weights
  for (i = 0; i < num_layers; i++) {
    device_weights[i] =
        subtract(device_weights[i], multiply(gradWeights[i], learning_rate));
  }
  
}

/* Helper Functions */

device_sample_t hostToDeviceSam(sample_t sample) {
  host_matrix_t host_x = sample[0];
  host_matrix_t host_y = sample[1];
  device_matrix_t dev_x;
  device_matrix_t dev_y;
  int i, j;
  for (i = 0; i < host_x.size(); i++) {
    device_vec_t row;
    for (j = 0; j < host_x[i].size(); j++) {
      row.push_back(host_x[i][j]);
    }
    dev_x.push_back(row);
  }
  for (i = 0; i < host_y.size(); i++) {
    device_vec_t row;
    for (j = 0; j < host_y[i].size(); j++) {
      row.push_back(host_y[i][j]);
    }
    dev_y.push_back(row);
  }
  device_sample_t dsam(dev_x, dev_y);
  return dsam; 
}

host_matrix_t deviceToHostWeight(device_matrix_t w) {
  //for every weight
  //copy device weight into the host weights
  return;
}

namespace dcnn {

void train(thrust::host_vector<sample_t> samples, int num_samples, 
           int input_rows, int input_cols, int output_rows, int output_cols,
           thrust::device_vector<int> num_units, 
           thrust::device_vector<layer_type_t> layer_types,
           int num_layers, double learning_rate, int num_epochs,
           thrust::host_vector<host_matrix_t> weights) 
{
  // iterators
  int e, s, i;
  thrust::device_vector<device_matrix_t> device_weights;
  thrust::device_vector<device_sample_t> device_samples;

  device_samples.resize(num_samples);
  // need to copy the samples from the host to the device
  for (s = 0; s < num_samples; s++) {
    device_samples[i] = hostToDeviceSam(samples[s]);
  }

  // alloc device_weights
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

    device_weights.push_back(device_init(rows, cols, 0.0));
  }

  // dimensions for the cuda blocks
  const int threadsPerBlock = 512;
  const int blocks = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

  for (e = 0; e < num_epochs; e++) {
    single_epoch_kernel<<<blocks, threadsPerBlock>>>
      (device_weights, device_samples, num_layers, learning_rate);
    cudaDeviceSynchronize();
  }

  // the device holds the correct set of weights
  // TODO: COPY THE WEIGHTS INTO THE HOST
  for (i = 0; i < num_layers; i++) {
    weights.push_back(deviceToHostWeight(device_weights[i]));
  }

}


size_t predict(thrust::host_vector<host_matrix_t> weights, host_matrix_t x, 
               int num_layers, thrust::host_vector<layer_type_t> layer_types) {
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
  for (int i = 0; i < comp.size(); i++) {
    if (comp[i][0] == maxVal) {
      return i;
    }
  }

  // should never reach this line
  return 0;
}

host_matrix_t linearForward(host_matrix_t a, host_matrix_t b) {
  // printf("In linearForward x/z then A/B \n");
  // display(a);
  // display(b);
  return dot(b, a);
}

host_matrix_t linearBackward1(host_matrix_t a, host_matrix_t b) {
  return dot(b, transpose(a));
}

host_matrix_t linearBackward2(host_matrix_t a, host_matrix_t b) {
  matrix_t trA = transpose(a);
  matrix_t slicedA = slice(trA, 1, trA.size());
  return dot(slicedA, b);
}

host_matrix_t sigmForward(host_matrix_t v) {
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

host_matrix_t sigmBackward(host_matrix_t linearComp, 
  host_matrix_t activationComp, host_matrix_t gradActivation) 
{
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

host_matrix_t tanhForward(host_matrix_t v)
{
  vec_t ones = init(1, 1.);
  host_matrix_t res = tanh(v);
  res.insert(res.begin(), ones);
  return res;
}

host_matrix_t model_t::tanhBackward(host_matrix_t linearComp, host_matrix_t activationComp,
                             host_matrix_t gradActivation) 
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  host_matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = gradActivation[i][j] * (1.0 - temp*temp);
    }
  }
  return res;
}

host_matrix_t model_t::reluForward(host_matrix_t A)
{
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  host_matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = std::max(0.0, A[i][j]);
    }
  }
  vec_t ones = init(1, 1.);
  B.insert(B.begin(), ones);
  return B;
}

host_matrix_t model_t::reluBackward(host_matrix_t linearComp, host_matrix_t activationComp,
                             host_matrix_t gradActivation) 
{
  size_t n = gradActivation.size();
  size_t m = gradActivation[0].size();
  host_matrix_t res = init(n, m, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      double temp = activationComp[i + 1][j];
      res[i][j] = temp == 0.0 ? 0 : gradActivation[i][j];
    }
  }
  return res;
}

host_matrix_t model_t::softForward(host_matrix_t v) {
  host_matrix_t exp_prev = exp(v);
  return divide(exp_prev, sum(exp_prev));
}

host_matrix_t model_t::softBackward(host_matrix_t y, host_matrix_t linearComp,
                             host_matrix_t activationComp, host_matrix_t gradActivation) {
  return subtract(activationComp, y);
}

double model_t::crossEntropyForward(host_matrix_t v, host_matrix_t vh) {
  host_matrix_t lvh = log(vh);
  double crossEntropy = 0.0;
  for (int i = 0; i < v.size(); i++) {
    crossEntropy -= v[i][0] * lvh[i][0];
  }
  return crossEntropy;
}

}; // namespace dcnn
