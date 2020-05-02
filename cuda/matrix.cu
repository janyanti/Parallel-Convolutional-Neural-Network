/*
  matrix.cu

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

*/

#include "math.h"
#include <limits>
// #include <random>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include "matrix.h"

namespace matrix {

/* initialize random seed for random functions */
// std::random_device rd;
// std::mt19937 rand_gen(rd());
// std::uniform_real_distribution<> rand_dist(0, 1);

thrust::minstd_rand device_rand_gen;
thrust::uniform_real_distribution<double> device_rand_dist(0, 1);

/*

*/
double sum(host_vec_t a) {
  double result = 0.0;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result += a[i];
  }

  return result;
}

/*

*/
double mean(host_vec_t a) {
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
double max(host_vec_t a) { return *std::max_element(a.begin(), a.end()); }

/*

*/
double min(host_vec_t a) { return *std::min_element(a.begin(), a.end()); }

/*

*/
host_vec_t log(host_vec_t a) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(math::log(a[i]));
  }

  return result;
}

/*

*/
host_vec_t exp(host_vec_t a) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(exp(a[i]));
  }

  return result;
}

/*

*/
host_vec_t tanh(host_vec_t a) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(tanh(a[i]));
  }

  return result;
}

/*

*/
host_vec_t pow(host_vec_t a, int exp) {
  size_t n = a.size();

  host_vec_t b;

  for (size_t i = 0; i < n; i++) {
    b.push_back(pow(a[i], exp));
  }

  return b;
}

/*

*/
host_vec_t multiply(host_vec_t a, double scalar) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(scalar * (a[i]));
  }

  return result;
}

/*

*/
host_vec_t divide(host_vec_t a, double scalar) {
  host_vec_t result;
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
double dot(host_vec_t a, host_vec_t b) {
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
host_vec_t add(host_vec_t a, host_vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector addition \n");
    exit(-1);
  }

  host_vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] + b[i]);
  }

  return c;
}

/*

*/
host_vec_t subtract(host_vec_t a, host_vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector subtract \n");
    exit(-1);
  }

  host_vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] - b[i]);
  }

  return c;
}

/*

*/
host_vec_t init(size_t n, double value) {
  assert(n > 0);
  host_vec_t a;

  for (size_t i = 0; i < n; i++) {
    a.push_back(value);
  }

  return a;
}

host_vec_t randu(size_t n) {
  host_vec_t a = init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    a[i] = device_rand_dist(device_rand_gen);
  }

  return a;
}

void display(host_vec_t a) {
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
double sum(host_matrix_t A) {
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
double mean(host_matrix_t A) {
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
double max(host_matrix_t A) {
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
double min(host_matrix_t A) {
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
host_matrix_t log(host_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  host_matrix_t result;

  host_matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = log(A[i][j]);
    }
  }

  return B;
}

/*

*/
host_matrix_t exp(host_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  host_matrix_t result;

  host_matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = exp(A[i][j]);
    }
  }

  return B;
}

/*

*/
host_matrix_t tanh(host_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  host_matrix_t B = init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = tanh(A[i][j]);
    }
  }

  return B;
}

/*

*/
host_matrix_t pow(host_matrix_t A, int exp) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  host_matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[i][j] = pow(A[i][j], exp);
    }
  }

  return B;
}

/*

*/
host_matrix_t transpose(host_matrix_t A) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  host_matrix_t B = init(m, n, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[j][i] = A[i][j];
    }
  }

  return B;
}

/*

*/
host_matrix_t multiply(host_matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  host_matrix_t B = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    host_vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      B[i][j] = scalar * (A[i][j]);
    }
  }

  return B;
}

/*

*/
host_matrix_t divide(host_matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  host_matrix_t B = init(n, m, 0.0);

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
host_matrix_t init(size_t n, size_t m, double value) {

  assert(n > 0 && m > 0);

  host_matrix_t A;

  for (size_t i = 0; i < n; i++) {
    host_vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      temp_n.push_back(value);
    }
    A.push_back(temp_n);
  }

  return A;
}

/*

*/
host_matrix_t randu(size_t n, size_t m) {
  assert(n > 0 && m > 0);

  host_matrix_t A = init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      A[i][j] = device_rand_dist(device_rand_gen);
    }
  }

  return A;
}

/*

*/
host_matrix_t add(host_matrix_t A, host_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for add\n");
  }

  host_matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

/*

*/
host_matrix_t subtract(host_matrix_t A, host_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for subtract\n");
  }

  host_matrix_t C = init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }

  return C;
}

/*

*/
host_matrix_t slice(host_matrix_t A, size_t start, size_t end) {
  size_t rows = end-start;
  size_t cols = A[0].size();
  host_matrix_t B;
  for (size_t i = 0; i < rows; i++) {
    host_vec_t v(A[start+i]);
    B.push_back(v);
  }
  return B;
}

/*

*/
host_matrix_t vector_to_matrix(host_vec_t a, size_t n, size_t m) {
  size_t len = a.size();

  if (n * m != len) {
    printf("Incorrect dimensions for vector reshape \n");
    exit(0);
  }

  host_matrix_t A = init(n, m, 0.0);

  for (size_t k = 0; k < len; k++) {
    size_t i = k / m;
    size_t j = k % m;
    A[i][j] = a[k];
  }

  return A;

}

/*

*/
host_matrix_t dot(host_matrix_t A, host_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (m_A != n_B) {
    printf("Incorrect matrix dimensions for dot product \n");
    exit(-1);
  }

  host_matrix_t C = init(n_A, m_B, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_B; j++) {
      for (size_t k = 0; k < m_A; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

void display(host_matrix_t A) {
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


/*

*/
inline __device__
double sum(device_vec_t a) {
  double result = 0.0;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result += a[i];
  }

  return result;
}

/*

*/
inline __device__
double mean(device_vec_t a) {
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
inline __device__
double max(device_vec_t a) { return *thrust::max_element(a.begin(), a.end()); }

/*

*/
inline __device__
double min(device_vec_t a) { return *thrust::min_element(a.begin(), a.end()); }

/*

*/
inline __device__
device_vec_t log(device_vec_t a) {
  device_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(log(a[i]));
  }

  return result;
}

/*

*/
inline __device__
device_vec_t exp(device_vec_t a) {
  device_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(exp(a[i]));
  }

  return result;
}

/*

*/
inline __device__
device_vec_t tanh(device_vec_t a) {
  device_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(tanh(a[i]));
  }

  return result;
}

/*

*/
inline __device__
device_vec_t pow(device_vec_t a, int exp) {
  size_t n = a.size();

  device_vec_t b;

  for (size_t i = 0; i < n; i++) {
    b.push_back(pow(a[i], exp));
  }

  return b;
}

/*

*/
inline __device__
device_vec_t multiply(device_vec_t a, double scalar) {
  device_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(scalar * (a[i]));
  }

  return result;
}

/*

*/
inline __device__
device_vec_t divide(device_vec_t a, double scalar) {
  device_vec_t result;
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
inline __device__
double dot(device_vec_t a, device_vec_t b) {
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
inline __device__
device_vec_t add(device_vec_t a, device_vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector addition \n");
    exit(-1);
  }

  device_vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] + b[i]);
  }

  return c;
}

/*

*/
inline __device__
device_vec_t subtract(device_vec_t a, device_vec_t b) {
  size_t n = a.size();
  size_t m = b.size();

  if (n != m) {
    printf("Incorrect dimensions for vector subtract \n");
    exit(-1);
  }

  device_vec_t c;

  for (size_t i = 0; i < n; i++) {
    c.push_back(a[i] - b[i]);
  }

  return c;
}

/*

*/
inline __device__
device_vec_t device_init(size_t n, double value) {
  assert(n > 0);
  device_vec_t a;

  for (size_t i = 0; i < n; i++) {
    a.push_back(value);
  }

  return a;
}

inline __device__
device_vec_t device_randu(size_t n) {
  device_vec_t a = device_init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    a[i] = device_rand_dist(device_rand_gen);
  }

  return a;
}

inline __device__
void display(device_vec_t a) {
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
inline __device__
double sum(device_matrix_t A) {
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
inline __device__
double mean(device_matrix_t A) {
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
inline __device__
double max(device_matrix_t A) {
  size_t n = A.size();
  double max, curr_val;
  max = std::numeric_limits<double>::min();

  for (size_t i = 0; i < n; i++) {
    curr_val = *thrust::max_element(A[i].begin(), A[i].end());

    if (curr_val > max) {
      max = curr_val;
    }
  }
  return max;
}

/*

*/
inline __device__
double min(device_matrix_t A) {
  size_t n = A.size();
  double min, curr_val;
  min = std::numeric_limits<double>::max();

  for (size_t i = 0; i < n; i++) {
    curr_val = *thrust::min_element(A[i].begin(), A[i].end());

    if (curr_val < min) {
      min = curr_val;
    }
  }
  return min;
}

/*

*/
inline __device__
device_matrix_t log(device_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  device_matrix_t result;

  device_matrix_t B = device_init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = thrust::log(A[i][j]);
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t exp(device_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  device_matrix_t result;

  device_matrix_t B = device_init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = thrust::exp(A[i][j]);
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t tanh(device_matrix_t A) {
  size_t ns = A.size();
  // Assert ns >= 1
  size_t ms = A[0].size();

  device_matrix_t B = device_init(ns, ms, 0.0);

  for (size_t i = 0; i < ns; i++) {
    for (size_t j = 0; j < ms; j++) {
      B[i][j] = thrust::tanh(A[i][j]);
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t pow(device_matrix_t A, int exp) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  device_matrix_t B = device_init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[i][j] = thrust::pow(A[i][j], exp);
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t transpose(device_matrix_t A) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  device_matrix_t B = device_init(m, n, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      B[j][i] = A[i][j];
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t multiply(device_matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  device_matrix_t B = device_init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    device_vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      B[i][j] = scalar * (A[i][j]);
    }
  }

  return B;
}

/*

*/
inline __device__
device_matrix_t divide(device_matrix_t A, double scalar) {
  size_t n = A.size();
  // Assert ns >= 1
  size_t m = A[0].size();

  device_matrix_t B = device_init(n, m, 0.0);

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
inline __device__
device_matrix_t device_init(size_t n, size_t m, double value) {

  assert(n > 0 && m > 0);

  device_matrix_t A;

  for (size_t i = 0; i < n; i++) {
    device_vec_t temp_n;
    for (size_t j = 0; j < m; j++) {
      temp_n.push_back(value);
    }
    A.push_back(temp_n);
  }

  return A;
}

/*

*/
inline __device__
device_matrix_t device_randu(size_t n, size_t m) {
  assert(n > 0 && m > 0);

  device_matrix_t A = device_init(n, m, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      A[i][j] = device_rand_dist(device_rand_gen);
    }
  }

  return A;
}

/*

*/
inline __device__
device_matrix_t add(device_matrix_t A, device_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for add\n");
  }

  device_matrix_t C = device_init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

/*

*/
inline __device__
device_matrix_t subtract(device_matrix_t A, device_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (n_A != n_B || m_A != m_B) {
    printf("Incorrect matrix dimensions for subtract\n");
  }

  device_matrix_t C = device_init(n_A, m_A, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_A; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }

  return C;
}

/*

*/
inline __device__
device_matrix_t slice(device_matrix_t A, size_t start, size_t end) {
  size_t rows = end-start;
  size_t cols = A[0].size();
  device_matrix_t B;
  for (size_t i = 0; i < rows; i++) {
    device_vec_t v(A[start+i]);
    B.push_back(v);
  }
  return B;
}

/*

*/
inline __device__
device_matrix_t vector_to_matrix(device_vec_t a, size_t n, size_t m) {
  size_t len = a.size();

  if (n * m != len) {
    printf("Incorrect dimensions for vector reshape \n");
    exit(0);
  }

  device_matrix_t A = device_init(n, m, 0.0);

  for (size_t k = 0; k < len; k++) {
    size_t i = k / m;
    size_t j = k % m;
    A[i][j] = a[k];
  }

  return A;

}

/*

*/
inline __device__
device_matrix_t dot(device_matrix_t A, device_matrix_t B) {
  size_t n_A = A.size();
  size_t m_A = A[0].size();
  size_t n_B = B.size();
  size_t m_B = B[0].size();

  if (m_A != n_B) {
    printf("Incorrect matrix dimensions for dot product \n");
    exit(-1);
  }

  device_matrix_t C = device_init(n_A, m_B, 0.0);

  for (size_t i = 0; i < n_A; i++) {
    for (size_t j = 0; j < m_B; j++) {
      for (size_t k = 0; k < m_A; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

inline __device__
void display(device_matrix_t A) {
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
