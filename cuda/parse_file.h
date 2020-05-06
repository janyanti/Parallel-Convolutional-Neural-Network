/*
  parse_file.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  File parser for the MNIST dataset
  Credits: mrgloom
           https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
*/
#pragma once

#include <vector>

#include "matrix.h"
#include "dcnn.h"

namespace pfile {

int reverse_int(int i);
matrix::matrix_t read_images(char *filepath);
matrix::matrix_t read_labels(char *filepath, int num_labels);
void create_sample(matrix::matrix_t data, matrix::matrix_t labels,
                   matrix::matrix_t *X,
                   matrix::matrix_t *Y);

}; // namespace pfile
