/*
  dcnn.h

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "matrix.h"

using namespace matrix;

namespace dcnn {

class sample_t {
private:
  host_matrix_t x;
  host_matrix_t y;

public:
  // Constructor
  sample_t (host_matrix_t inputX, host_matrix_t inputY) {
     x = inputX;
     y = inputY;
  }

  host_matrix_t getX() {
    return x;
  }

  host_matrix_t getY() {
    return y;
  }
};

class device_sample_t {
private:
  device_matrix_t x;
  device_matrix_t y;

public:
  // Constructor
  device_sample_t (device_matrix_t inputX, device_matrix_t inputY) {
     x = inputX;
     y = inputY;
  }

  device_matrix_t getX() {
    return x;
  }

  device_matrix_t getY() {
    return y;
  }
};

enum layer_type_t {
  SIGM,
  SOFT,
  TANH,
  RELU,
  CONV
};


class model_t {
private:
  int num_layers;
  double learning_rate;
  int num_epochs;
  thrust::host_vector<int> num_units;
  thrust::host_vector<layer_type_t> layer_types;
  thrust::host_vector<host_matrix_t> weights;

  thrust::device_vector<int> cuda_num_units;
  thrust::device_vector<layer_type_t> cuda_layer_types;
  thrust::device_vector<device_matrix_t> cuda_weights;
  thrust::device_vector<device_sample_t> device_samples;

public:
  // Constructor
  model_t (int numL, thrust::host_vector<int> numU,
           thrust::host_vector<layer_type_t> ltypes, double lr, int e) {

    num_layers = numL;
    num_units = numU;
    layer_types = ltypes;
    learning_rate = lr;
    num_epochs = e;

    cuda_num_units.resize(numL);
    cuda_layer_types.resize(numL);
    thrust::copy(num_units.begin(), num_units.end(), cuda_num_units.begin());
    thrust::copy(layer_types.begin(), layer_types.end(), cuda_layer_types.begin());

  }

  void train(thrust::host_vector<sample_t> samples, int num_samples, int input_rows, int input_cols,
             int output_rows, int output_cols);
  size_t predict(host_matrix_t x);
  host_matrix_t linearForward(host_matrix_t a, host_matrix_t b);
  host_matrix_t linearBackward1(host_matrix_t a, host_matrix_t b);
  host_matrix_t linearBackward2(host_matrix_t a, host_matrix_t b);
  host_matrix_t sigmForward(host_matrix_t v);
  host_matrix_t sigmBackward(host_matrix_t linearComp, host_matrix_t activationComp,
                        host_matrix_t gradActivation);
  host_matrix_t softForward(host_matrix_t v);
  host_matrix_t softBackward(host_matrix_t y, host_matrix_t linearComp,
                        host_matrix_t activationComp, host_matrix_t gradActivation);
  double crossEntropyForward(host_matrix_t v, host_matrix_t vh);
  host_matrix_t tanhForward(host_matrix_t v);
  host_matrix_t tanhBackward(host_matrix_t linearComp, host_matrix_t activationComp,
                        host_matrix_t gradActivation);
  host_matrix_t reluForward(host_matrix_t v);
  host_matrix_t reluBackward(host_matrix_t linearComp, host_matrix_t activationComp,
                        host_matrix_t gradActivation);
  // double crossEntropyBackward();
};

} // namespace dcnn
