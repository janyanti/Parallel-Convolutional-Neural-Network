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
private:
  matrix_t x;
  matrix_t y;
public:
  // Constructor
  sample_t (matrix_t inputX, matrix_t inputY) {
     x = inputX;
     y = inputY;
  }

  matrix_t getX() {
    return x;
  }

  matrix_t getY() {
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
  std::vector<int> num_units;
  std::vector<layer_type_t> layer_types;
  double learning_rate;
  int num_epochs;

public:
  // Constructor 
  model_t (int numL, std::vector<int> numU, 
           std::vector<layer_type_t> ltypes, double lr, int e) {
    num_layers = numL;
    num_units = numU;
    layer_types = ltypes;
    learning_rate = lr;
    num_epochs = e;
  }

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
  matrix_t tanhForward(matrix_t v);
  matrix_t tanhBackward(matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);
  matrix_t reluForward(matrix_t v);
  matrix_t reluBackward(matrix_t linearComp, matrix_t activationComp,
                        matrix_t gradActivation);
  // double crossEntropyBackward();
};

} // namespace dcnn
