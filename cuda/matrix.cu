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
    result.push_back(::log(a[i]));
  }

  return result;
}

/*

*/
host_vec_t exp(host_vec_t a) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(::exp(a[i]));
  }

  return result;
}

/*

*/
host_vec_t tanh(host_vec_t a) {
  host_vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(::tanh(a[i]));
  }

  return result;
}

/*

*/
host_vec_t pow(host_vec_t a, int exp) {
  size_t n = a.size();

  host_vec_t b;

  for (size_t i = 0; i < n; i++) {
    b.push_back(::pow(a[i], exp));
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
      B[i][j] = ::log(A[i][j]);
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
      B[i][j] = ::exp(A[i][j]);
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
      B[i][j] = ::tanh(A[i][j]);
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
      B[i][j] = ::pow(A[i][j], exp);
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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*
// called by the host with dims <1,1>
*/
__global__
void sum(size_t n, double* a, double* result) {
  // double result = 0.0;
  // size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    *result += a[i];
  }

  return;
}

/*
// called by the host with dims <1,1>
*/
// __global__
// void mean(device_vec_t a, double* result) {
//   double total = sum(a);
//   double length = (double)a.size();
//   // double result = 0.0;

//   if (length > 0.0) {
//     *result = *result + total / length;
//   }

//   return;
// }

/*
// called by the host with dims <1,1>
*/
__global__
void max(size_t n, double* a, double* result) { 
  *result = a[0];
  for (int i = 0; i < n; i++) {
    if (a[i] > *result) *result = a[i];
  }
}

/*
// called by the host with dims <1,1>
*/
__global__
void min(size_t n, double* a, double* result) {
  *result = a[0];
  for (int i = 0; i < n; i++) {
    if (a[i] < *result) *result = a[i];
  }
}

/*

*/
__global__
void log(size_t n, double* a, double* b) {
  // device_vec_t result;
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  b[i] = ::log(a[i]);
    // result.push_back();
  // }

  return;
}

/*

*/
__global__
void exp(size_t n, double* a, double* b) {
  // device_vec_t result;
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  b[i] = ::exp(a[i]);
    // result.push_back();
  // }

  return;
}

/*

*/
__global__
void tanh(size_t n, double* a, double* b) {
  // device_vec_t result;
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  b[i] = ::tanh(a[i]);
    // result.push_back(::tanh(a[i]));
  // }

  return;
}

/*

*/
__global__
void pow(size_t n, double* a, int exp, double* b) {
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // device_vec_t b;

  // for (size_t i = 0; i < n; i++) {
  b[i] = ::pow(a[i], exp);
    // b.push_back(::pow(a[i], exp));
  // }

  return;
}

/*

*/
__global__
void multiply(size_t n, double* a, double scalar, double* b) {
  // device_vec_t result;
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  b[i] = scalar * a[i];
    // result.push_back(scalar * (a[i]));
  // }

  return;
}

/*

*/
__global__
void divide(size_t n, double* a, double scalar, double* b) {
  // device_vec_t result;
  // size_t n = a.size();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  // for (size_t i = 0; i < n; i++) {
  b[i] = a[i]/ scalar;
    // result.push_back(a[i] / scalar);
  // }

  return;
}

/*

*/
__global__
void dot(size_t n, double* a, double* b, double* result) {
  // size_t n = a.size();
  // size_t m = b.size();

  *result = 0.0;

  // if (n != m) {
  //   printf("Incorrect dimensions for vector dot product\n");
  //   exit(-1);
  // }

  for (size_t i = 0; i < n; i++) {
    *result += a[i] * b[i];
  }

  return;
}

/*

*/
__global__
void add(size_t n, double* a, double* b, double* c) {
  // size_t n = a.size();
  // size_t m = b.size();

  // if (n != m || m != c.size()) {
  //   printf("Incorrect dimensions for vector addition \n");
  //   exit(-1);
  // }

  // device_vec_t c;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  c[i] = (a[i] + b[i]);
  // }

  return;
}

/*

*/
__global__
void subtract(size_t n, double* a, double* b, double* c) {
  // size_t n = a.size();
  // size_t m = b.size();

  // if (n != m || m != c.size()) {
  //   printf("Incorrect dimensions for vector subtract \n");
  //   exit(-1);
  // }

  // device_vec_t c;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // for (size_t i = 0; i < n; i++) {
  c[i] = (a[i] - b[i]);
  // }

  return;
}

/*

*/
device_vec_t device_init(size_t n, double value) {
  assert(n > 0);
  device_vec_t a;

  a.resize(0);

  for (size_t i = 0; i < n; i++) {
    a.push_back(value);
  }

  return a;
}

device_vec_t device_randu(size_t n) {
  device_vec_t a = device_init(n, 0.0);

  for (size_t i = 0; i < n; i++) {
    a[i] = device_rand_dist(device_rand_gen);
  }

  return a;
}

/*
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
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*

*/
__global__
void sum(size_t n, size_t m, double** A, double* result) {
  // size_t ns = A.size();
  // Assert ns >= 1
  // size_t ms = A.data().get()[0].size();
  *result = 0.0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      *result += A[i][j];
    }
  }

  return;
}

/*

*/
// __global__
// double mean(device_matrix_t A) {
//   size_t ns = A.size();
//   // Assert ns >= 1
//   size_t ms = (A.data().get()[0]).size();
//   double total = sum(A);
//   double length = (double)(ns * ms);
//   double result = 0.0;

//   if (length > 0.0) {
//     result = total / length;
//   }

//   return result;
// }

/*

*/
__global__
void max(size_t n, size_t m, double** A, double* result) {
  *result = A[0][0];
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      if (A[i][j] > *result) *result = A[i][j];
    }
  }
}

/*

*/
__global__
void min(size_t n, size_t m, double** A, double* result) {
  *result = A[0][0];
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      if (A[i][j] < *result) *result = A[i][j];
    }
  }
}

/*

*/
__global__
void log(size_t n, size_t m, double** A, double** B) {
  // size_t ns = A.size();
  // Assert ns >= 1
  // size_t ms = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;

  // device_matrix_t B = device_init(ns, ms, 0.0);

  // for (size_t i = 0; i < ns; i++) {
  //   for (size_t j = 0; j < ms; j++) {
  B[i][j] = ::log(A[i][j]);
  //   }
  // }

  return;
}

/*

*/
__global__
void exp(size_t n, size_t m, double** A, double** B) {
  // size_t ns = A.size();
  // Assert ns >= 1
  // size_t ms = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;
  // device_matrix_t B = device_init(ns, ms, 0.0);

  // for (size_t i = 0; i < ns; i++) {
  //   for (size_t j = 0; j < ms; j++) {
  B[i][j] = ::exp(A[i][j]);
  //   }
  // }

  return;
}

/*

*/
__global__
void tanh(size_t n, size_t m, double** A, double** B) {
  // size_t ns = A.size();
  // Assert ns >= 1
  // size_t ms = A.data().get()[0].size();

  // device_matrix_t B = device_init(ns, ms, 0.0);
  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;
  // for (size_t i = 0; i < ns; i++) {
  //   for (size_t j = 0; j < ms; j++) {
  B[i][j] = ::tanh(A[i][j]);
  //   }
  // }

  return;
}

/*

*/
__global__
void pow(size_t n, size_t m, double** A, int exp, double** B) {
  // size_t n = A.size();
  // Assert ns >= 1
  // size_t m = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;
  // device_matrix_t B = device_init(n, m, 0.0);

  // for (size_t i = 0; i < n; i++) {
  //   for (size_t j = 0; j < m; j++) {
  B[i][j] = ::pow(A[i][j], exp);
  //   }
  // }

  return;
}

/*

*/
__global__
void transpose(size_t n, size_t m, double** A, double** B) {
  // size_t n = A.size();
  // Assert ns >= 1
  // size_t m = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;
  // device_matrix_t B = device_init(m, n, 0.0);

  // for (size_t i = 0; i < n; i++) {
  //   for (size_t j = 0; j < m; j++) {
  B[j][i] = A[i][j];
  //   }
  // }

  return;
}

/*

*/
__global__
void multiply(size_t n, size_t m, double** A, double scalar, double** B) {
  // size_t n = A.size();
  // Assert ns >= 1
  // size_t m = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;

  // device_matrix_t B = device_init(n, m, 0.0);

  // for (size_t i = 0; i < n; i++) {
  //   for (size_t j = 0; j < m; j++) {
  B[i][j] = scalar * (A[i][j]);
  //   }
  // }

  return;
}

/*

*/
__global__
void divide(size_t n, size_t m, double** A, double scalar, double** B) {
  // size_t n = A.size();
  // Assert ns >= 1
  // size_t m = A.data().get()[0].size();

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= m) return;
  // device_matrix_t B = device_init(n, m, 0.0);

  // if (scalar == 0.0) {
  //   printf("Attempt to divide by zero \n");
  // }

  // for (size_t i = 0; i < n; i++) {
  //   for (size_t j = 0; j < m; j++) {
  B[i][j] = A[i][j] / scalar;
  //   }
  // }

  return;
}

/*

*/
__host__ 
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
__host__ 
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
__global__
void add(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C) {
  // size_t n_A = A.size();
  // size_t m_A = A.data().get()[0].size();
  // size_t n_B = B.size();
  // size_t m_B = B.data().get()[0].size();

  // if (n_A != n_B || m_A != m_B) {
  //   printf("Incorrect matrix dimensions for add\n");
  // }
  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_A || j >= m_A) return;
   // = device_init(n_A, m_A, 0.0);

  // for (size_t i = 0; i < n_A; i++) {
  //   for (size_t j = 0; j < m_A; j++) {
  C[i][j] = A[i][j] + B[i][j];
  //   }
  // }

  return ;
}

/*

*/
__global__
void subtract(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C) {
  // size_t n_A = A.size();
  // size_t m_A = A.data().get()[0].size();
  // size_t n_B = B.size();
  // size_t m_B = B.data().get()[0].size();

  // if (n_A != n_B || m_A != m_B) {
  //   printf("Incorrect matrix dimensions for subtract\n");
  // }

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_A || j >= m_A) return;

  // device_matrix_t C = device_init(n_A, m_A, 0.0);

  // for (size_t i = 0; i < n_A; i++) {
  //   for (size_t j = 0; j < m_A; j++) {
  C[i][j] = A[i][j] - B[i][j];
  //   }
  // }

  return;
}

/*

*/
__host__ 
device_matrix_t slice(device_matrix_t A, size_t start, size_t end) {
  size_t rows = end-start;
  size_t cols = A.data().get()[0].size();
  device_matrix_t B;
  for (size_t i = 0; i < rows; i++) {
    device_vec_t v(A[start+i]);
    B.push_back(v);
  }
  return B;
}

/*

*/
__host__ 
device_matrix_t vector_to_matrix(device_vec_t a, size_t n, size_t m) {
  size_t len = a.size();

  // if (n * m != len) {
    // printf("Incorrect dimensions for vector reshape \n");
    // exit(0);
  // }

  device_matrix_t A = device_init(n, m, 0.0);

  for (size_t k = 0; k < len; k++) {
    size_t i = k / m;
    size_t j = k % m;
    A.data().get()[i][j] = a[k];
  }

  return A;

}

/*

*/
__global__
void dot(size_t n_A, size_t m_A, size_t n_B, size_t m_B, double** A, double** B, double** C) {
  // size_t n_A = A.size();
  // size_t m_A = A.data().get()[0].size();
  // size_t n_B = B.size();
  // size_t m_B = B.data().get()[0].size();

  // if (m_A != n_B) {
  //   printf("Incorrect matrix dimensions for dot product \n");
  //   exit(-1);
  // }

  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_A || j >= m_B) return;

  // for (size_t i = 0; i < n_A; i++) {
  //   for (size_t j = 0; j < m_B; j++) {
  for (size_t k = 0; k < m_A; k++) {
    C[i][j] += A[i][k] * B[k][j];
  }
  //   }
  // }

  return;
}

/*
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
} */

} // namespace matrix
