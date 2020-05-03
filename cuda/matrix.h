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
#include "math.h"

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
__global__ void sum(size_t n, double* a, double* result);   // called by the host with dims <1,1>
// __global__ void mean(device_vec_t a, double* result);    // called by the host with dims <1,1>
__global__ void max(size_t n, double* a, double* result);   // called by the host with dims <1,1>
__global__ void min(size_t n, double* a, double* result);   // called by the host with dims <1,1>
__global__ void log(size_t n, double* a, double* b);
__global__ void exp(size_t n, double* a, double* b);
__global__ void tanh(size_t n, double* a, double* b);
__global__ void pow(size_t n, double* a, int exp, double* b);
__global__ void multiply(size_t n, double* a, double scalar, double* b);
__global__ void divide(size_t n, double* a, double scalar, double* b);
__global__ void dot(size_t n, double* a, double* b, double* result);   // called by the host with dims <1,1>
__global__ void add(size_t n, double* a, double* b, double* c);
__global__ void subtract(size_t n, double* a, double* b, double* c);
device_vec_t device_init(size_t n, double value);
device_vec_t device_randu(size_t n);

device_matrix_t slice(device_matrix_t A, size_t n, size_t m);
__global__ void dot(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double* A, double* B, double* C);
__global__ void transpose(size_t n, size_t m, double* A, double* B);
// inline __device__ void display(device_vec_t a);

/* device Matrix functions */
// __global__ void sum(size_t n, size_t m, double** A, double* result);   // called by the host with dims <1,1>
// __global__ void mean(device_matrix_t A, double* result);  // called by the host with dims <1,1>
// __global__ void max(size_t n, size_t m, double** A, double* result);   // called by the host with dims <1,1>
// __global__ void min(size_t n, size_t m, double** A, double* result);   // called by the host with dims <1,1>
// __global__ void log(size_t n, size_t m, double** A, double** B);
// __global__ void exp(size_t n, size_t m, double** A, double** B);
// __global__ void tanh(size_t n, size_t m, double** A, double** B);
// __global__ void pow(size_t n, size_t m, double** A, int exp, double** B);
// __global__ void multiply(size_t n, size_t m, double** A, double scalar, double** B);
// __global__ void divide(size_t n, size_t m, double** A, double scalar, double** B);
// __global__ void dot(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C);
// __global__ void add(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C);
// __global__ void subtract(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C);
// 
// __host__ device_matrix_t vector_to_matrix(device_vec_t a, size_t n, size_t m);
// __host__ device_matrix_t device_init(size_t n, size_t m, double value);
// __host__ device_matrix_t device_randu(size_t n, size_t m);
// inline __device__ void display(device_matrix_t A);

} // namespace matrix
