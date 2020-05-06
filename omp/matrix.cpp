/*
  matrix.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

*/

#include <cmath>
#include <limits>
#include <random>

#include "matrix.h"

namespace matrix {

/* initialize random seed for random functions */
std::random_device rd;
std::mt19937 rand_gen(rd());
std::uniform_real_distribution<> rand_dist(-1, 1);

/*

*/
double sum(vec_t a) {
  double result = 0.0;
  size_t n = a->n;

  for (size_t i = 0; i < n; i++) {
    result += a->data[i];
  }

  return result;
}

/*

*/
double mean(vec_t a) {
  double total = sum(a);
  double length = (double)a->n;
  double result = 0.0;

  if (length > 0.0) {
    result = total / length;
  }

  return result;
}

/*

*/
double max(vec_t a) {
  double res = a->data[0];
  for (size_t i = 0; i < a->n; i++) {
    if (a->data[i] > res )
      res = a->data[i];
  }
  return res;
}

/*

*/
double min(vec_t a) {
  double res = a->data[0];
  for (size_t i = 0; i < a->n; i++) {
    if (a->data[i] < res )
      res = a->data[i];
  }
  return res;
}

/*

*/
vec_t log(vec_t a) {

  size_t n = a->n;
  vec_t result = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    result->data[i] = (std::log(a->data[i]));
  }

  return result;
}

/*

*/
vec_t exp(vec_t a) {

  size_t n = a->n;
  vec_t result = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    result->data[i] = (std::exp(a->data[i]));
  }

  return result;
}

/*

*/
vec_t tanh(vec_t a) {
  size_t n = a->n;
  vec_t result = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    result->data[i] = (std::tanh(a->data[i]));
  }

  return result;
}

/*

*/
vec_t pow(vec_t a, int exp) {
  size_t n = a->n;

  vec_t result = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    result->data[i] = (std::pow(a->data[i], exp));
  }

  return result;
}

/*

*/
vec_t multiply(vec_t a, double scalar) {
  size_t n = a->n;
  vec_t result = init(n, 0.0);


  for (size_t i = 0; i < n; i++) {
    result->data[i] = (scalar * (a->data[i]));
  }

  return result;
}

/*

*/
vec_t divide(vec_t a, double scalar) {
  size_t n = a->n;
  vec_t result = init(n, 0.0);

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  for (size_t i = 0; i < n; i++) {
    result->data[i] = (a->data[i] / scalar);
  }

  return result;
}

/*

*/
double dot(vec_t a, vec_t b) {
  size_t n = a->n;
  size_t m = b->n;

  double result = 0.0;

  if (n != m) {
    printf("Incorrect dimensions for vector dot product\n");
    exit(-1);
  }

  for (size_t i = 0; i < n; i++) {
    result += a->data[i] * b->data[i];
  }

  return result;
}

/*

*/
vec_t add(vec_t a, vec_t b) {
  size_t n = a->n;
  size_t m = b->n;

  if (n != m) {
    printf("Incorrect dimensions for vector addition \n");
    exit(-1);
  }

  vec_t c = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    c->data[i] = a->data[i] + b->data[i];
  }

  return c;
}

/*

*/
vec_t subtract(vec_t a, vec_t b) {
  size_t n = a->n;
  size_t m = b->n;

  if (n != m) {
    printf("Incorrect dimensions for vector subtract \n");
    exit(-1);
  }

  vec_t c = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    c->data[i] = a->data[i] - b->data[i];
  }

  return c;
}

/*

*/
vec_t init(size_t n, double value) {
  assert(n > 0);

  vec_t a = (vec_t)calloc(sizeof(vec), 1);
  if (a == NULL) {
    fprintf(stderr, "bad alloc of size %ld LINE: %d \n", sizeof(vec),
    __LINE__);
    exit(-1);
  }

  a->data = (double*)calloc(sizeof(double), n);
  if (a->data == NULL) {
    fprintf(stderr, "bad alloc of size %ld LINE: %d \n", sizeof(double) * n,
    __LINE__);
    exit(-1);
  }

  a->n = n;

  for (size_t i = 0; i < n; i++) {
    a->data[i] = value;
  }

  return a;
}

vec_t randu(size_t n) {
  vec_t a = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    a->data[i] = rand_dist(rand_gen);
  }

  return a;
}

void display(vec_t a) {
  size_t n = a->n;

  printf("[ ");
  for (size_t i = 0; i < n; i++) {
    if (i == n - 1) {
      printf("%lf ] \n", a->data[i]);
    } else {
      printf("%lf, ", a->data[i]);
    }
  }
}

void clear(vec_t a) {
  memset(&a,0,sizeof(double)*a->n);
}

void vec_free(vec_t a) {
  free(a->data);
  free(a);
}

/*

*/
double sum(matrix_t A) {
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;
  double result = 0.0;

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      result += A->data[i][j];
    }
  }

  return result;
}

/*

*/
double mean(matrix_t A) {
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;
  double total = sum(A);
  double length = (double)(ns * ms);
  double result = 0.0;

  if (length > 0.0) {
    result = total / length;
  }

  return result;
}

/*

*/
double max(matrix_t A) {
  size_t n = A->n;
  size_t m = A->m;
  double max, curr_val;
  max = std::numeric_limits<double>::min();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      curr_val = A->data[i][j];
      if (curr_val > max) {
        max = curr_val;
      }
    }
  }
  return max;
}

/*

*/
double min(matrix_t A) {
  size_t n = A->n;
  size_t m = A->m;
  double min, curr_val;
  min = std::numeric_limits<double>::max();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      curr_val = A->data[i][j];
      if (curr_val < min) {
        min = curr_val;
      }
    }
  }
  return min;
}

/*

*/
matrix_t log(matrix_t A) {
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;

  matrix_t result;

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B->data[i][j] = std::log(A->data[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t exp(matrix_t A) {
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;

  matrix_t result;

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B->data[i][j] = std::exp(A->data[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t tanh(matrix_t A) {
  size_t ns = A->n;
  // Assert ns >= 1
  size_t ms = A->m;

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B->data[i][j] = std::tanh(A->data[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t pow(matrix_t A, int exp) {
  size_t n = A->n;
  // Assert ns >= 1
  size_t m = A->m;

  matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B->data[i][j] = std::pow(A->data[i][j], exp);
    }
  }

  return B;
}

/*

*/
matrix_t transpose(matrix_t A) {
  size_t n = A->n;
  // Assert ns >= 1
  size_t m = A->m;

  matrix_t B = init(m, n, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B->data[j][i] = A->data[i][j];
    }
  }

  return B;
}

/*

*/
matrix_t multiply(matrix_t A, double scalar) {
  size_t n = A->n;
  // Assert ns >= 1
  size_t m = A->m;

  matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      B->data[i][j] = scalar * (A->data[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t divide(matrix_t A, double scalar) {
  size_t n = A->n;
  // Assert ns >= 1
  size_t m = A->m;

  matrix_t B = init(n, m, 0.0);

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B->data[i][j] = A->data[i][j] / scalar;
    }
  }

  return B;
}

/*

*/
matrix_t init(size_t n, size_t m, double value) {

  assert(n > 0 && m > 0);

  matrix_t A = (matrix_t)calloc(sizeof(matrix),1);
  if (A == NULL) {
    fprintf(stderr, "bad alloc of size %ld LINE: %d \n", sizeof(matrix),
    __LINE__);
    exit(-1);
  }

  A->n = n;
  A->m = m;

  A->data = (double**)calloc(sizeof(double*), n);
  if (A->data == NULL) {
    fprintf(stderr, "bad alloc of size %ld LINE: %d \n", sizeof(double) * n,
    __LINE__);
    exit(-1);
  }

  for (size_t i = 0; i < n; i++) {
    A->data[i] = (double*)calloc(sizeof(double), m);
    if (A->data[i] == NULL) {
      fprintf(stderr, "bad alloc of size %ld LINE: %d \n", sizeof(double) * m,
      __LINE__);
      exit(-1);
    }
    for (size_t j = 0; j < m; j++) {
      A->data[i][j] = value;
    }
  }

  return A;
}

/*

*/
matrix_t randu(size_t n, size_t m) {
  assert(n > 0 && m > 0);

  matrix_t A = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      A->data[i][j] = rand_dist(rand_gen);
    }
  }

  return A;
}

/*

*/
matrix_t add(matrix_t A, matrix_t B) {
  size_t n_A = A->n;
  size_t m_A = A->m;
  size_t n_B = B->n;
  size_t m_B = B->m;

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for add\n");
  }

  matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C->data[i][j] = A->data[i][j] + B->data[i][j];
    }
  }

  return C;
}

/*

*/
matrix_t subtract(matrix_t A, matrix_t B) {
  size_t n_A = A->n;
  size_t m_A = A->m;
  size_t n_B = B->n;
  size_t m_B = B->m;

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for subtract\n");
  }

  matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C->data[i][j] = A->data[i][j] - B->data[i][j];
    }
  }

  return C;
}

/*

*/
matrix_t slice(matrix_t A, size_t start, size_t end) {
  size_t rows = end-start;
  size_t cols = A->m;
  matrix_t B = init(rows, cols, 0.0);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      B->data[i][j] = A->data[start+i][j];
    }
  }
  return B;
}

/*

*/
matrix_t vector_to_matrix(vec_t a, size_t n, size_t m) {
  size_t len = a->n;

  if (n * m != len) {
    printf("Incorrect dimensions for vector reshape \n");
    exit(0);
  }

  matrix_t A = init(n, m, 0.0);

  for (size_t k = 0; k < len; k++) {
    size_t i = k / m;
    size_t j = k % m;
    A->data[i][j] = a->data[k];
  }

  return A;

}

/*

*/
matrix_t dot(matrix_t A, matrix_t B) {
  size_t n_A = A->n;
  size_t m_A = A->m;
  size_t n_B = B->n;
  size_t m_B = B->m;

  if (m_A != n_B) {
    printf("Incorrect matrix dimensions for dot product \n");
    exit(-1);
  }

  matrix_t C = init(n_A, m_B, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_B; j++) {
      for (size_t k = 0; k < m_A; k++) {
        C->data[i][j] += A->data[i][k] * B->data[k][j];
      }
    }
  }

  return C;
}

void display(matrix_t A) {
  size_t n = A->n;
  size_t m = A->m;

  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("[ ");
    for (size_t j = 0; j < m; j++) {
      if (j == m - 1) {
        printf("%lf ", A->data[i][j]);
      } else {
        printf("%lf, ", A->data[i][j]);
      }
    }
    if (i == n - 1) {
      printf("]");
    } else {
      printf("] \n");
    }
  }
  printf("] \n\n");
}

void clear(matrix_t A) {
  size_t n = A->n;

  for (size_t i = 0; i < n; i++) {
    memset(&A->data[i], 0, sizeof(double) * A->m);
  }

}

void matrix_free(matrix_t A) {
  size_t n = A->n;

  for (size_t i = 0; i < n; i++) {
    free(A->data[i]);
  }

  free(A->data);
  free(A);
}

} // namespace matrix
