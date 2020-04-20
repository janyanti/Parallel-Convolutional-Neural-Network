/*
  matrix.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Simple Linear Algerbra Libraray for Machine
  TODO: Description of library functions
*/

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "matrix.h"
#include "dcnn.h"

using namespace matrix;
using namespace dcnn;

void matrix_test() {
  /* vector tests */
  size_t n = 5;
  size_t m = 5;

  matrix_t A0 = init(n, m, 0.0);
  matrix_t A1 = init(n, m, 1.0);
  matrix_t A2 = randu(n, m);

  display(A0);
  display(A1);
  display(A2);
  printf("SUM 1: %lf \n", sum(A1));
  printf("SUM 2: %lf \n", sum(A2));
  printf("MEAN 1: %lf \n", mean(A1));
  printf("MEAN 2: %lf \n", mean(A2));
  printf("MAX: %lf \n", max(A2));
  printf("MIN: %lf \n", min(A2));
  printf("LOG: ");
  display(log(A2));
  printf("\n  EXP: ");
  display(exp(A2));
  printf("\n  TANH: ");
  display(tanh(A2));
  printf("\n  POW 2: ");
  display(pow(A2, 2));
  printf("\n  TRANS: ");
  display(transpose(A2));

}

void nn_test() {
  std::vector<int> unitcounts = {5, 10};
  std::vector<layer_type_t> layers = {SIGM, SOFT};
  model_t m(2, unitcounts, layers, .1, 1);

  int numsamples = 2;
  int inputRows = 5;
  int inputCols = 1;
  int outputRows = 10;
  int outputCols = 1;

  matrix_t x1, y1;
  x1 = init(inputRows + 1, inputCols, 0.0);
  x1[0][0] = 1.0;
  x1[1+1][0] = 1.0;
  x1[2+1][0] = 1.0;
  x1[3+1][0] = 1.0;
  y1 = init(outputRows, outputCols, 0.0);
  y1[7][0] = 1.0;
  sample_t s1(x1, y1);

  matrix_t x2, y2;
  x2 = init(inputRows + 1, inputCols, 0.0);
  x2[0][0] = 1.0;
  x2[3+1][0] = 1.0;
  y2 = init(outputRows, outputCols, 0.0);
  y2[9][0] = 1.0;
  sample_t s2(x2, y2);

  std::vector<sample_t> samples = {s1, s2};
  
  m.train(samples, numsamples, inputRows, inputCols, outputRows, outputCols);
  return;
}


int main(int argc, char** argv) {

  //matrix_test();
  nn_test();
  return 0;

}
