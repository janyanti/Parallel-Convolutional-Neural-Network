/*
  dcnn.h

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Deep Convolutional neural network training model
  TODO:
*/

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "matrix.h"

#define BATCH_SIZE 8

using namespace matrix;

namespace dcnn {

enum layer_type_t {
  SIGM,
  SOFT,
  TANH,
  RELU,
  CONV
};


  void train(std::vector<matrix_t> &X, std::vector<matrix_t> &Y, int num_samples,
             int input_rows, int input_cols, int output_rows, int output_cols,
             std::vector<int> num_units, std::vector<layer_type_t> layer_types,
             int num_layers, double learning_rate, int num_epochs,
             std::vector<matrix_t> &weights);
  size_t predict(std::vector<matrix_t> &weights, matrix_t &x,
                 int num_layers, std::vector<layer_type_t> layer_types);
  void linearForward(matrix_t dest, matrix_t a, matrix_t b);
  void linearBackward1(matrix_t dest, matrix_t a, matrix_t b);
  void linearBackward2(matrix_t dest, matrix_t a, matrix_t b);
  void sigmForward(matrix_t dest, matrix_t v);
  void sigmBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);
  void softForward(matrix_t dest, matrix_t v);
  void softBackward(matrix_t dest, matrix_t y, matrix_t linearComp,
                        matrix_t activationComp, matrix_t gradActivation);
  double crossEntropyForward(matrix_t v, matrix_t vh);
  void tanhForward(matrix_t dest, matrix_t v);
  void tanhBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);
  void reluForward(matrix_t dest, matrix_t v);
  void reluBackward(matrix_t dest, matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);

} // namespace dcnn
