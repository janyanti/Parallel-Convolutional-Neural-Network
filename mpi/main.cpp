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
#include "instrument.h"
#include "comms.h"

using namespace matrix;
using namespace dcnn;

void real_test(char **argv, int process_count, int process_id) {



  bool mpi_master = process_id == 0;


  START_ACTIVITY(ACTIVITY_SETUP);
  /* Initialize variables for file parsing */
  matrix_t train_data;
  matrix_t train_labels;
  matrix_t test_data;
  matrix_t test_labels;

  int num_labels = 10;
  std::vector<matrix_t> X_train;
  std::vector<matrix_t> Y_train;
  std::vector<matrix_t> X_test;
  std::vector<matrix_t> Y_test;

  pfile::read_images(argv[1], train_data);
  pfile::read_labels(argv[2], num_labels, train_labels);

  pfile::read_images(argv[3], test_data);
  pfile::read_labels(argv[4], num_labels, test_labels);

  /* Initialize training/testing samples */
  pfile::create_sample(train_data, train_labels, X_train, Y_train);
  pfile::create_sample(test_data, test_labels, X_test, Y_test);

  /* Initialize NN parameter */
  int numsamples = X_train.size();
  int inputRows = X_train[0].size();
  int inputCols = 1;
  int outputRows = num_labels;
  int outputCols = 1;

  std::vector<int> unitcounts = {16, 12, outputRows};
  std::vector<layer_type_t> layers = {TANH, TANH, SOFT};
  size_t num_layers = layers.size();
  double learning_rate = 0.001;
  int num_epochs = 100;

  std::vector<matrix_t> weights;

  fprintf(stderr, "Training on machine: %d \n", process_id);

  FINISH_ACTIVITY(ACTIVITY_SETUP);

  START_ACTIVITY(ACTIVITY_PROP);
  train(X_train, Y_train, numsamples, inputRows, inputCols,
        outputRows, outputCols, unitcounts, layers,
        num_layers, learning_rate, num_epochs, weights);

  FINISH_ACTIVITY(ACTIVITY_PROP);

  /* Evaluate model performance */
  START_ACTIVITY(ACTIVITY_PREDICT);
  double correct = 0;
  for (size_t i = 0; i < X_test.size(); i++){
    matrix_t X = X_test[i];
    matrix_t Y = Y_test[i];
    size_t yh_index = predict(weights, X, num_layers, layers);
    if (Y[yh_index][0] == 1.0)
      correct++;
  }

  double error_rate = correct / ((double) X_test.size());
  printf("Accuracy: %% %.2f\n", 100. * error_rate);

  FINISH_ACTIVITY(ACTIVITY_PREDICT);
}

int main(int argc, char **argv) {

  if (argc < 5) {
    fprintf(stdout, "Incorrect number of parameters\n");
  }


  /* Initialize MPI Environment */
  int process_count = 1;
  int process_id = 0;
  int num_processes = 0;

#if MPI
  MPI_Init(NULL, NULL);
  process_count = MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  process_id = MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
#endif

  /* Enable instrumentation */
  TRACK_ACTIVITY(true);

  /* Run NN test with command line arguments */
  real_test(argv, process_count, process_id);

  /* Export instrumenation data */
  SHOW_ACTIVITY(stderr);

#if MPI
  MPI_Finalize();
#endif

  return 0;
}
