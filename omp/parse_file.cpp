/*
  parse_file.h

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  File parser for the MNIST dataset
  Credits: mrgloom
           https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
*/
#include "parse_file.h"
#include <fstream>
#include <iostream>

#define NUM_IMAGES 1

namespace pfile {

int reverse_int(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 0xFF;
  ch2 = (i >> 8) & 0xFF;
  ch3 = (i >> 16) & 0xFF;
  ch4 = (i >> 24) & 0xFF;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

matrix::matrix_t read_images(char *filepath) {
  std::ifstream file(filepath, std::ios::binary);

  if (file.is_open()) {
    int magic_number = 0;
    int num_images = 0;
    int data_image = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char *)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);
    data_image = n_rows * n_cols;

    matrix_t arr = init(NUM_IMAGES, data_image, 0.0);

    for (int i = 0; i < NUM_IMAGES; ++i) {
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          arr->data[i][(n_rows * r) + c] = (double)temp / 255.0;
        }
      }
    }
    return arr;
  }

  /* Error opening file */
  return NULL;
}

matrix::matrix_t read_labels(char *filepath, int num_labels) {

  std::ifstream file(filepath, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int num_images = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char *)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);


    matrix_t arr = init(NUM_IMAGES, num_labels, 0.0);

    for (int i = 0; i < NUM_IMAGES; ++i) {
      unsigned char j = 0;
      file.read((char *)&j, sizeof(j));
      arr->data[i][j] = 1.0;
    }

    return arr;
  }

  /* Error opening file */
  return NULL;

}

void create_sample(matrix::matrix_t data, matrix::matrix_t labels,
                   matrix::matrix_t *X, matrix::matrix_t *Y) {

  size_t n = data->n;
  size_t m = labels->n;

  size_t num_data = data->m;
  size_t num_labels = labels->m;

  if (n != m) {
    printf("Data for training and labels don't match \n");
  }

  for (size_t i = 0; i < n; i++) {
    size_t j;
    matrix::matrix_t data_matrix = init(num_data + 1, 1, 1.0);
    for (j = 0; j < num_data; j++)
      data_matrix->data[j+1][0] = data->data[i][j];

    matrix::matrix_t label_matrix = init(num_labels, 1, 1.0);
    for (j = 0; j < num_labels; j++)
      label_matrix->data[j][0] = labels->data[i][j];

    X[i] = data_matrix;
    Y[i] = label_matrix;
  }

}

}; // namespace pfile
