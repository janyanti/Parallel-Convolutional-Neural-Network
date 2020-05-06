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
//#include "instrument.h"


using namespace matrix;
using namespace dcnn;

void real_test(char **argv) {

  //START_ACTIVITY(ACTIVITY_SETUP);

  /* Initialize variables for file parsing */

  int num_labels = 10;

  //TODO: Check that pointers are not NULL
  matrix_t train_data = pfile::read_images(argv[1]);
  matrix_t train_labels = pfile::read_labels(argv[2], num_labels);

  matrix_t test_data = pfile::read_images(argv[3]);
  matrix_t test_labels = pfile::read_labels(argv[4], num_labels);

  /* Initialize training/testing samples */
  matrix_t* X_train = (matrix_t*)calloc(sizeof(matrix_t), train_data->n);
  matrix_t* Y_train = (matrix_t*)calloc(sizeof(matrix_t), train_labels->n);;
  matrix_t* X_test = (matrix_t*)calloc(sizeof(matrix_t), test_data->n);
  matrix_t* Y_test = (matrix_t*)calloc(sizeof(matrix_t), test_labels->n);

  pfile::create_sample(train_data, train_labels, X_train, Y_train);
  pfile::create_sample(test_data, test_labels, X_test, Y_test);

  /* Initialize NN parameter */
  int numsamples = train_data->n;
  int inputRows = X_train[0]->n;
  int inputCols = X_train[0]->m;
  int outputRows = Y_train[0]->n;
  int outputCols = Y_train[0]->m;

  int* unitcounts = (int*)calloc(sizeof(int), 3);
  unitcounts[0] = 16;
  unitcounts[1] = 12;
  unitcounts[2] = outputRows;

  layer_type_t* layers = (layer_type_t*)calloc(sizeof(layer_type_t), 3);
  layers[0] = TANH;
  layers[1] = TANH;
  layers[2] = SOFT;

  size_t num_layers = 3;
  double learning_rate = 0.00075;
  int num_epochs = 5;

  //FINISH_ACTIVITY(ACTIVITY_SETUP);

  //START_ACTIVITY(ACTIVITY_TRAIN);

  matrix_t* weights = train(X_train, Y_train, numsamples, inputRows, inputCols,
                            outputRows, outputCols, unitcounts, layers,
                            num_layers, learning_rate, num_epochs);

  //FINISH_ACTIVITY(ACTIVITY_TRAIN);

  /* Evaluate model performance */
  //START_ACTIVITY(ACTIVITY_PREDICT);

  // test after training
  double correct = 0;
  for (size_t i = 0; i < test_data->n; i++){
    matrix_t X = X_test[i];
    matrix_t Y = Y_test[i];
    size_t yh_index = predict(weights, X, unitcounts, num_layers, layers);
    // printf("%d %f\n", yh_index, Y->data[yh_index][0]);
    size_t idx = yh_index * test_data->m;
    if (Y->data[idx] == 1.0)
      correct++;
  }

  double error_rate = correct / ((double)test_data->n);
  printf("Accuracy: %% %.2f \n", 100. * error_rate);

  //FINISH_ACTIVITY(ACTIVITY_PREDICT);



  for (size_t i = 0; i < train_data->n; i++) {
    matrix_free(X_train[i]);
    matrix_free(Y_train[i]);
  }

  free(X_train);
  free(Y_train);

  for (size_t i = 0; i < test_data->n; i++) {
    matrix_free(X_test[i]);
    matrix_free(Y_test[i]);
  }

  free(X_test);
  free(Y_test);

  matrix_free(train_data);
  matrix_free(test_data);
  matrix_free(train_labels);
  matrix_free(test_labels);

  for(size_t i = 0; i < num_layers; i++) {
    matrix_free(weights[i]);
  }
  free(weights);
  free(unitcounts);
  free(layers);



}

int main(int argc, char **argv) {

  if (argc < 5) {
    fprintf(stdout, "Incorrect number of parameters\n");
  }

  /* Enable instrumentation */
  //TRACK_ACTIVITY(true);

  real_test(argv);

  /* Export instrumenation data */
  //SHOW_ACTIVITY(stderr);

  return 0;
}
