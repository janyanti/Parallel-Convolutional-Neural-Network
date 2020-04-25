/*
  matrix.h

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Simple Linear Algerbra Libraray for Machine
  TODO: Description of library functions
*/
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace matrix {
/* Host Vector and Matrix Type declarations */
typedef thrust::host_vector<double> host_vec_t;
typedef thrust::host_vector<host_vec_t> host_matrix_t;

/* Host Vector functions */
double sum(host_vec_t a);
double mean(host_vec_t a);
double max(host_vec_t a);
double min(host_vec_t a);
host_vec_t log(host_vec_t a);
host_vec_t exp(host_vec_t a);
host_vec_t tanh(host_vec_t a);
host_vec_t pow(host_vec_t a, int exp);
host_vec_t multiply(host_vec_t a, double scalar);
host_vec_t divide(host_vec_t a, double scalar);
double dot(host_vec_t a, host_vec_t b);
host_vec_t add(host_vec_t a, host_vec_t b);
host_vec_t subtract(host_vec_t a, host_vec_t b);
host_vec_t init(size_t n, double value);
host_vec_t randu(size_t n);
void display(host_vec_t a);

/* Host Matrix functions */
double sum(host_matrix_t A);
double mean(host_matrix_t A);
double max(host_matrix_t A);
double min(host_matrix_t A);
host_matrix_t log(host_matrix_t A);
host_matrix_t exp(host_matrix_t A);
host_matrix_t tanh(host_matrix_t A);
host_matrix_t pow(host_matrix_t A, int exp);
host_matrix_t transpose(host_matrix_t A);
host_matrix_t multiply(host_matrix_t A, double scalar);
host_matrix_t divide(host_matrix_t A, double scalar);
host_matrix_t dot(host_matrix_t A, host_matrix_t B);
host_matrix_t add(host_matrix_t A, host_matrix_t B);
host_matrix_t subtract(host_matrix_t A, host_matrix_t B);
host_matrix_t slice(host_matrix_t A, size_t n, size_t m);
host_matrix_t vector_to_matrix(host_vec_t a, size_t n, size_t m);
host_matrix_t init(size_t n, size_t m, double value);
host_matrix_t randu(size_t n, size_t m);
void display(host_matrix_t A);

/* Device Vector and Matrix Type declarations */
typedef thrust::device_vector<double> device_vec_t;
typedef thrust::device_vector<device_vec_t> device_matrix_t;

/* device Vector functions */
double sum(device_vec_t a);
double mean(device_vec_t a);
double max(device_vec_t a);
double min(device_vec_t a);
device_vec_t log(device_vec_t a);
device_vec_t exp(device_vec_t a);
device_vec_t tanh(device_vec_t a);
device_vec_t pow(device_vec_t a, int exp);
device_vec_t multiply(device_vec_t a, double scalar);
device_vec_t divide(device_vec_t a, double scalar);
double dot(device_vec_t a, device_vec_t b);
device_vec_t add(device_vec_t a, device_vec_t b);
device_vec_t subtract(device_vec_t a, device_vec_t b);
device_vec_t init(size_t n, double value);
device_vec_t randu(size_t n);
void display(device_vec_t a);

/* device Matrix functions */
double sum(device_matrix_t A);
double mean(device_matrix_t A);
double max(device_matrix_t A);
double min(device_matrix_t A);
device_matrix_t log(device_matrix_t A);
device_matrix_t exp(device_matrix_t A);
device_matrix_t tanh(device_matrix_t A);
device_matrix_t pow(device_matrix_t A, int exp);
device_matrix_t transpose(device_matrix_t A);
device_matrix_t multiply(device_matrix_t A, double scalar);
device_matrix_t divide(device_matrix_t A, double scalar);
device_matrix_t dot(device_matrix_t A, device_matrix_t B);
device_matrix_t add(device_matrix_t A, device_matrix_t B);
device_matrix_t subtract(device_matrix_t A, device_matrix_t B);
device_matrix_t slice(device_matrix_t A, size_t n, size_t m);
device_matrix_t vector_to_matrix(device_vec_t a, size_t n, size_t m);
device_matrix_t init(size_t n, size_t m, double value);
device_matrix_t randu(size_t n, size_t m);
void display(device_matrix_t A);

} // namespace matrix
