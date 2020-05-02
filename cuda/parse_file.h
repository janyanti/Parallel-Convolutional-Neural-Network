/*
  parse_file.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  File parser for the MNIST dataset
  Credits: mrgloom
           https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
*/
#include <vector>

#include "matrix.h"
#include "dcnn.h"

namespace pfile {

int reverse_int(int i);
void read_images(char *filepath, matrix::host_matrix_t &arr);
void read_labels(char *filepath, int num_labels, matrix::host_matrix_t &arr);
thrust::host_vector<dcnn::sample_t> create_sample(matrix::host_matrix_t data, matrix::host_matrix_t labels);

}; // namespace pfile

