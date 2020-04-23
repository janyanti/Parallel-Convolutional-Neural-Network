/*
  matrix.c

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
std::uniform_real_distribution<> rand_dist(0, 1);

/*

*/
double sum(vec_t a) {
  double result = 0.0;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result += a[i];
  }

  return result;
}

/*

*/
double mean(vec_t a) {
  double total = sum(a);
  double length = (double)a.size();
  double result = 0.0;

  if (length > 0.0) {
    result = total / length;
  }

  return result;
}

/*

*/
double max(vec_t a) { return *std::max_element(a.begin(), a.end()); }

/*

*/
double min(vec_t a) { return *std::min_element(a.begin(), a.end()); }

/*

*/
vec_t log(vec_t a) {
  vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(std::log(a[i]));
  }

  return result;
}

/*

*/
vec_t exp(vec_t a) {
  vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(std::exp(a[i]));
  }

  return result;
}

/*

*/
vec_t tanh(vec_t a) {
  vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(std::tanh(a[i]));
  }

  return result;
}

/*

*/
vec_t pow(vec_t a, int exp) {
  size_t n = a.size();

  vec_t b;

  for (size_t i = 0; i < n; i++) {
    b.push_back(std::pow(a[i], exp));
  }

  return b;
}

/*

*/
vec_t multiply(vec_t a, double scalar) {
  vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(scalar * (a[i]));
  }

  return result;
}

/*

*/
vec_t divide(vec_t a, double scalar) {
  vec_t result;
  size_t n = a.size();

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  for (size_t i = 0; i < n; i++) {
    result.push_back(a[i] / scalar);
  }

  return result;
}

/*

*/
double dot(vec_t a, vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  double result = 0.0;

  if (n != m) {
    printf("Incorrect dimensions for vector dot product\n");
    exit(-1);
  }

  for (size_t i = 0; i < n; i++) {
    result += a[i] * b[i];
  }

  return result;
}

/*

*/
vec_t add(vec_t a, vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector addition \n");
    exit(-1);
  }

  vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] + b[i]);
  }

  return c;
}

/*

*/
vec_t subtract(vec_t a, vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector subtract \n");
    exit(-1);
  }

  vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] - b[i]);
  }

  return c;
}

/*

*/
vec_t init(size_t n, double value) {
  assert(n > 0);
  vec_t a;

  for (size_t i = 0; i < n; i++) {
    a.push_back(value);
  }

  return a;
}

vec_t randu(size_t n) {
  vec_t a = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    a[i] = rand_dist(rand_gen);
  }

  return a;
}

void display(vec_t a) {
  size_t n = a.size();

  printf("[ ");
  for (size_t i = 0; i < n; i++) {
    if (i == n - 1) {
      printf("%lf ] \n", a[i]);
    } else {
      printf("%lf, ", a[i]);
    }
  }
}

/*

*/
double sum(matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();
  double result = 0.0;

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      result += A[i][j];
    }
  }

  return result;
}

/*

*/
double mean(matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();
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
  size_t n = A.size();
  double max, curr_val;
  max = std::numeric_limits<double>::min();

  for (size_t i = 0; i < n; i++) {
    curr_val = *std::max_element(A[i].begin(), A[i].end());

    if (curr_val > max) {
      max = curr_val;
    }
  }
  return max;
}

/*

*/
double min(matrix_t A) {
  size_t n = A.size();
  double min, curr_val;
  min = std::numeric_limits<double>::max();

  for (size_t i = 0; i < n; i++) {
    curr_val = *std::min_element(A[i].begin(), A[i].end());

    if (curr_val < min) {
      min = curr_val;
    }
  }
  return min;
}

/*

*/
matrix_t log(matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  matrix_t result;

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = std::log(A[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t exp(matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  matrix_t result;

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = std::exp(A[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t tanh(matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = std::tanh(A[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t pow(matrix_t A, int exp) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[i][j] = std::pow(A[i][j], exp);
    }
  }

  return B;
}

/*

*/
matrix_t transpose(matrix_t A) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  matrix_t B = init(m, n, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[j][i] = A[i][j];
    }
  }

  return B;
}

/*

*/
matrix_t multiply(matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      B[i][j] = scalar * (A[i][j]);
    }
  }

  return B;
}

/*

*/
matrix_t divide(matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  matrix_t B = init(n, m, 0.0);

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[i][j] = A[i][j] / scalar;
    }
  }

  return B;
}

/*

*/
matrix_t init(size_t n, size_t m, double value) {

  assert(n > 0 && m > 0);

  matrix_t A;

  for (size_t i = 0; i < n; i++) {
    vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      temp_n.push_back(value);
    }
    A.push_back(temp_n);
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
      A[i][j] = rand_dist(rand_gen);
    }
  }

  return A;
}

/*

*/
matrix_t add(matrix_t A, matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for add\n");
  }

  matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

/*

*/
matrix_t subtract(matrix_t A, matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for subtract\n");
  }

  matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }

  return C;
}

/*

*/
matrix_t slice(matrix_t A, size_t start, size_t end) {
  size_t rows = end-start;
  size_t cols = A[0].size();
  matrix_t B;
  for (size_t i = 0; i < rows; i++) {
    vec_t v(A[start+i]);
    B.push_back(v);
  }
  return B;
}

/*

*/
matrix_t vector_to_matrix(vec_t a, size_t n, size_t m) {
  size_t len = a.size();

  if (n * m != k) {
    printf("Incorrect dimensions for vector reshape \n");
    exit(0);
  }

  matrix_t A = init(n, m, 0.0);

  for (size_t k = 0; k < len; i++) {
    size_t i = k / m;
    size_t j = k % m;
    A[r][c] = a[k];
  }

}

/*

*/
matrix_t dot(matrix_t A, matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (m_A != n_B) {
    printf("Incorrect matrix dimensions for dot product \n");
    exit(-1);
  }

  matrix_t C = init(n_A, m_B, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_B; j++) {
      for (size_t k = 0; k < m_A; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

void display(matrix_t A) {
  size_t n = A.size();
  size_t m = A[0].size();

  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("[ ");
    for (size_t j = 0; j < m; j++) {
      if (j == m - 1) {
        printf("%lf ", A[i][j]);
      } else {
        printf("%lf, ", A[i][j]);
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

} // namespace matrix
