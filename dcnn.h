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

using namespace matrix;

namespace dcnn {

class sample_t {
public:
  matrix_t x;
  matrix_t y;
};

enum layer_type {
  SIGM,
  SOFT,
  TANH,
  RELU,
  CONV
};

class model {
private:
  int num_layers;
  int *num_units;
  layer_type *layer_types;
  double learning_rate;
  int num_epochs;

public:
  void train(std::vector<sample_t> samples, int num_samples, int input_rows, int input_cols,
             int output_rows, int output_cols);
  matrix_t linearForward(matrix_t a, matrix_t b);
  matrix_t linearBackward1(matrix_t a, matrix_t b);
  matrix_t linearBackward2(matrix_t a, matrix_t b);
  matrix_t sigmForward(matrix_t v);
  matrix_t sigmBackward(matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);
  matrix_t softForward(matrix_t v);
  matrix_t softBackward(matrix_t y, matrix_t linearComp,
                        matrix_t activationComp, matrix_t gradActivation);
  double crossEntropyForward(matrix_t v, matrix_t vh);
  // matrix_t tanhForward(matrix_t v);
  // matrix_t tanhBackward();
  // matrix_t reluForward(matrix_t v);
  // matrix_t reluBackward();
  // double crossEntropyBackward();
};

} // namespace dcnn
