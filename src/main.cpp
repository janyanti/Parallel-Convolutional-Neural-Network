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

void nn_test() {


  int numsamples = 2;
  int inputRows = 5;
  int inputCols = 1;
  int outputRows = 10;
  int outputCols = 1;

  std::vector<int> unitcounts = {5, 8, outputRows};
  std::vector<layer_type_t> layers = {SIGM, SIGM, SOFT};
  model_t m(3, unitcounts, layers, .1, 50);

  matrix_t x1, y1;
  x1 = init(inputRows + 1, inputCols, 0.0);
  x1->data[0][0] = 1.0;
  x1->data[1+1][0] = 1.0;
  x1->data[2+1][0] = 1.0;
  x1->data[3+1][0] = 1.0;
  y1 = init(outputRows, outputCols, 0.0);
  y1->data[7][0] = 1.0;
  sample_t s1(x1, y1);

  matrix_t x2, y2;
  x2 = init(inputRows + 1, inputCols, 0.0);
  x2->data[0][0] = 1.0;
  x2->data[3+1][0] = 1.0;
  y2 = init(outputRows, outputCols, 0.0);
  y2->data[9][0] = 1.0;
  sample_t s2(x2, y2);

  std::vector<sample_t> samples = {s1, s2};

  m.train(samples, numsamples, inputRows, inputCols, outputRows, outputCols);
  return;
}

void pfile_test(char **argv) {
  int num_labels = 10;

  matrix_t train_data = pfile::read_images(argv[1]);
  matrix_t train_labels = pfile::read_labels(argv[2], num_labels);

  //display(train_labels->data[0]);

}

void real_test(char **argv) {
  // read in the data here
  int num_labels = 10;
  std::vector<sample_t> train_samples;
  std::vector<sample_t> test_samples;

  matrix_t train_data = pfile::read_images(argv[1]);
  matrix_t train_labels = pfile::read_labels(argv[2], num_labels);

  matrix_t test_data = pfile::read_images(argv[3]);
  matrix_t test_labels = pfile::read_labels(argv[4], num_labels);

  train_samples = pfile::create_sample(train_data, train_labels);
  test_samples = pfile::create_sample(test_data, test_labels);

  int numsamples = train_samples.size();
  int inputRows = train_data->m;
  int inputCols = 1;
  int outputRows = num_labels;
  int outputCols = 1;

  // init the NN
  std::vector<int> unitcounts = {28, 12, outputRows};
  std::vector<layer_type_t> layers = {TANH, TANH, SOFT};
  size_t num_layers = layers.size();
  model_t m(num_layers, unitcounts, layers, .00075, 75);
  m.train(train_samples, numsamples, inputRows, inputCols, outputRows, outputCols);

  // test after training
  double correct;
  for (int i = 0; i < test_samples.size(); i++){
    sample_t s = test_samples[i];
    size_t yh_index = m.predict(s.getX());
    if (s.getY()->data[yh_index][0] == 1.0)
      correct++;
  }

  double error_rate = correct / ((double) test_samples.size());
  printf("Accuracy: %% %.2f\n", 100. * error_rate);

}

int main(int argc, char **argv) {

  if (argc < 5) {
    fprintf(stdout, "Incorrect number of parameters\n");
  }

  real_test(argv);

  return 0;
}
