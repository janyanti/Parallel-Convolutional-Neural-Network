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

void real_test(char **argv) {
  // read in the data here
  host_matrix_t train_data;
  host_matrix_t train_labels;
  host_matrix_t test_data;
  host_matrix_t test_labels;

  int num_labels = 10;
  std::vector<sample_t> train_samples;
  std::vector<sample_t> test_samples;

  pfile::read_images(argv[1], train_data);
  pfile::read_labels(argv[2], num_labels, train_labels);

  pfile::read_images(argv[3], test_data);
  pfile::read_labels(argv[4], num_labels, test_labels);

  train_samples = pfile::create_sample(train_data, train_labels);
  test_samples = pfile::create_sample(test_data, test_labels);

  int numsamples = train_samples.size();
  int inputRows = train_data[0].size();
  int inputCols = 1;
  int outputRows = num_labels;
  int outputCols = 1;

  // init the NN
  std::vector<int> unitcounts = {5, 8, outputRows};
  std::vector<layer_type_t> layers = {SIGM, SIGM, SOFT};
  model_t m(3, unitcounts, layers, .001, 75);
  m.train(train_samples, numsamples, inputRows, inputCols, outputRows, outputCols);

  // test after training
  double correct;
  for (int i = 0; i < test_samples.size(); i++){
    sample_t s = test_samples[i];
    size_t yh_index = m.predict(s.getX());
    if (s.getY()[yh_index][0] == 1.0)
      correct++;
  }

  double error_rate = correct / ((double) test_samples.size());
  printf("%f\n", error_rate);

}

int main(int argc, char **argv) {

  if (argc < 5) {
    fprintf(stdout, "Incorrect number of parameters\n");
  }

  real_test(argv);

  return 0;
}
