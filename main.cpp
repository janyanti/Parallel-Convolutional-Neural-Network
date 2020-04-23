/*
  matrix.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Simple Linear Algerbra Libraray for Machine
  TODO: Description of library functions
*/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "matrix.h"
#include "dcnn.h"
#include "parse_file.h"

using namespace matrix;
using namespace dcnn;

void matrix_test() {
  /* vector tests */
  size_t n = 5;
  size_t m = 5;
  size_t k = 40;

  matrix_t A0 = init(n, m, 0.0);
  matrix_t A1 = init(n, m, 1.0);
  matrix_t A2 = randu(n, m);
  matrix_t A3 = randu(m, n);

  vec_t a = init(k, 1.0);

  display(A0);
  display(A1);
  display(A2);
  display(A3);

  display(dot(A2, A3));
  display(dot(A3, A2));

  matrix_t B1 = vector_to_matrix(a, 5,8);
  matrix_t B2 = vector_to_matrix(a, 8,5);

  display(dot(B2, B1));
  display(dot(B1, B2));


}

void nn_test(char **argv) {
  int num_labels = 10;

  matrix_t train_data;
  matrix_t train_labels;

  pfile::read_images(argv[1], train_data);
  pfile::read_labels(argv[2], num_labels, train_labels);

  std::vector<sample_t> mnist;

  for (size_t i = 0; i < train_data.size(); i++) {
    mnist.push_back(sample_t(train_data[i], train_labels[i]));
  }

  std::vector<int> unitcounts = {5, 10};
  std::vector<layer_type_t> layers = {SIGM, SOFT};

  model_t m(2, unitcounts, layers, 0.01, 1);

  int numsamples = 60000;
  int inputRows = 5;
  int inputCols = 1;
  int outputRows = 10;
  int outputCols = 1;
}

int main(int argc, char **argv) {

   matrix_test();
  // if (argc < 3) {
  //   fprintf(stdout, "Incorrect number of parameters\n");
  // }

  return 0;
}
