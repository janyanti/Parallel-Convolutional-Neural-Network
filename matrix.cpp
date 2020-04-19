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
double max(vec_t a) {
  return *std::max_element(a.begin(), a.end());
}


/*

*/
double min(vec_t a) {
  return *std::min_element(a.begin(), a.end());
}


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
vec_t init(vec_t a, double value) {
  vec_t result;
  size_t n = a.size();

  for (size_t i = 0; i < n; i++) {
    result.push_back(value);
  }

  return result;
}


/*

*/
double sum(matrix_t A) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();
  double result = 0.0;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result += A[i][j];
    }
  }

  return result;
}

/*

*/
double mean(matrix_t A) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();
  double total = sum(A);
  double length = (double)(rows * cols);
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
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();

  matrix_t result;

  for (size_t i = 0; i < rows; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < cols; j++) {
      temp_row.push_back(std::log(A[i][j]));
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t exp(matrix_t A) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();

  matrix_t result;

  for (size_t i = 0; i < rows; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < cols; j++) {
      temp_row.push_back(std::exp(A[i][j]));
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t tanh(matrix_t A) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();

  matrix_t result;

  for (size_t i = 0; i < rows; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < cols; j++) {
      temp_row.push_back(std::tanh(A[i][j]));
    }
    result.push_back(temp_row);
  }

  return result;

}

/*

*/
matrix_t multiply(matrix_t A, double scalar) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();

  matrix_t result;

  for (size_t i = 0; i < rows; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < cols; j++) {
      temp_row.push_back(scalar * (A[i][j]));
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t divide(matrix_t A, double scalar) {
  size_t rows = A.size();
  // Assert rows >= 1
  size_t cols = A[0].size();

  matrix_t result;

  if (scalar == 0.0) {
    printf("Attempt to divide by zero \n");
  }

  for (size_t i = 0; i < rows; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < cols; j++) {
      temp_row.push_back((A[i][j]) / scalar);
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t init(size_t n, size_t m, double value) {

  assert(n > 0 && m > 0);

  matrix_t result;

  for (size_t i = 0; i < n; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < m; j++) {
      temp_row.push_back(value);
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t randu(size_t n, size_t m) {
  assert(n > 0 && m > 0);

  matrix_t result;

  for (size_t i = 0; i < n; i++) {
    vec_t temp_row;
    for (size_t j = 0; j < m; j++) {
      double value = rand_dist(rand_gen);
      temp_row.push_back(value);
    }
    result.push_back(temp_row);
  }

  return result;
}

/*

*/
matrix_t dot(matrix_t A, matrix_t B) {
  size_t row_A = A.size();
  size_t col_A = A[0].size();
  size_t row_B = B.size();
  size_t col_B = B[0].size();

  if (col_A != row_B) {
    printf("Incorrect matrix dimensions for dot product \n");
    exit(-1);
  }

  matrix_t C = init(row_A, col_B, 0.0);

  for (size_t i = 0; i < row_A; i++) {\
    for (size_t j = 0; j < col_B; j++) {
      for (size_t k = 0; k < col_A; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

} // namespace matrix
