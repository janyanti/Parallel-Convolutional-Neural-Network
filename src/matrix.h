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
#include <string.h>
#include <vector>

namespace matrix {
/* Vector and Matrix Type declarations */
typedef std::vector<double> vec_t;
typedef std::vector<vec_t> matrix_t;

/* Vector functions */
double sum(vec_t &a);
double mean(vec_t &a);
double max(vec_t &a);
double min(vec_t &a);
vec_t log(vec_t &a);
vec_t exp(vec_t &a);
vec_t tanh(vec_t &a);
vec_t pow(vec_t &a, int exp);
vec_t multiply(vec_t &a, double scalar);
vec_t divide(vec_t &a, double scalar);
double dot(vec_t &a, vec_t &b);
vec_t add(vec_t &a, vec_t &b);
vec_t subtract(vec_t &a, vec_t &b);
vec_t init(size_t n, double value);
vec_t randu(size_t n);
void display(vec_t &a);
void clear(vec_t &a);


/* Vector functions with destructive interface */
void log(vec_t &src, vec_t &dest);
void exp(vec_t &src, vec_t &dest);
void tanh(vec_t &src, vec_t &dest);
void pow(vec_t &src, vec_t &dest, int exp);
void multiply(vec_t &src, vec_t &dest, double scalar);
void divide(vec_t &src, vec_t &dest, double scalar);
void add(vec_t &srcA, vec_t &srcB, vec_t &dest);
void subtract(vec_t &srcA, vec_t &srcB, vec_t &dest);

/* Matrix functions */
double sum(matrix_t &A);
double mean(matrix_t &A);
double max(matrix_t &A);
double min(matrix_t &A);
matrix_t log(matrix_t &A);
matrix_t exp(matrix_t &A);
matrix_t tanh(matrix_t &A);
matrix_t pow(matrix_t &A, int exp);
matrix_t transpose(matrix_t &A);
matrix_t multiply(matrix_t &A, double scalar);
matrix_t divide(matrix_t &A, double scalar);
matrix_t dot(matrix_t &A, matrix_t &B);
matrix_t add(matrix_t &A, matrix_t &B);
matrix_t subtract(matrix_t &A, matrix_t &B);
matrix_t slice(matrix_t &A, size_t n, size_t m);
matrix_t vector_to_matrix(vec_t &a, size_t n, size_t m);
matrix_t init(size_t n, size_t m, double value);
matrix_t randu(size_t n, size_t m);
void display(matrix_t &A);
void clear(matrix_t &A);

/* Matrix functions with destructive interface */
void log(matrix_t &src, matrix_t &dest);
void exp(matrix_t &src, matrix_t &dest);
void tanh(matrix_t &src, matrix_t &dest);
void pow(matrix_t &src, matrix_t &dest, int exp);
void transpose(matrix_t &src, matrix_t &dest);
void multiply(matrix_t &src, matrix_t &dest, double scalar);
void divide(matrix_t &src, matrix_t &dest, double scalar);
void dot(matrix_t &srcA, matrix_t &srcB, matrix_t &dest);
void add(matrix_t &srcA, matrix_t &srcB, matrix_t &dest);
void subtract(matrix_t &srcA, matrix_t &srcB, matrix_t &dest);


} // namespace matrix
