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

typedef thrust::host_vector<host_matrix_t> sample_t;
typedef thrust::device_vector<device_matrix_t> device_sample_t;

enum layer_type_t {
  SIGM,
  SOFT,
  TANH,
  RELU,
  CONV
};

void train(thrust::host_vector<sample_t> samples, int num_samples, 
           int input_rows, int input_cols, int output_rows, int output_cols,
           thrust::device_vector<int> num_units, 
           thrust::device_vector<layer_type_t> layer_types,
           int num_layers, double learning_rate, int num_epochs, 
           thrust::host_vector<host_matrix_t> weights);
size_t predict(thrust::host_vector<host_matrix_t> weights, host_matrix_t x, 
               int num_layers, thrust::host_vector<layer_type_t> layer_types);
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


} // namespace dcnn
