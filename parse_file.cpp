/*
  parse_file.h

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  File parser for the MNIST dataset
  Credits: mrgloom
           https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
*/
#include "parse_file.h"
#include "matrix.h"
#include <fstream>
#include <iostream>

namespace pfile {

int reverse_int(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_images(char *filepath, matrix::matrix_t &arr) {
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
    arr.resize(num_images, matrix::vec_t(data_image));

    for (int i = 0; i < num_images; ++i) {
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          arr[i][(n_rows * r) + c] = (double)temp;
        }
      }
    }
  }
}

void read_labels(char *filepath, int num_labels,
                 matrix::matrix_t &arr) {

  std::ifstream file(filepath, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int num_images = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char *)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);
    arr.resize(num_images, matrix::vec_t(num_labels));

    for (int i = 0; i < num_images; ++i) {
      unsigned char j = 0;
      file.read((char *)&j, sizeof(j));
      arr[i][j] = 1.0;
    }
  }
}

}; // namespace pfile
