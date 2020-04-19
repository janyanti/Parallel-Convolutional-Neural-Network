/*
  matrix.cpp

  Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)

  Simple Linear Algerbra Libraray for Machine
  TODO: Description of library functions
*/

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "matrix.h"

using namespace matrix;

void matrix_test() {
  /* vector tests */
  size_t n = 5;
  size_t m = 5;

  matrix_t A0 = init(n, m, 0.0);
  matrix_t A1 = init(n, m, 1.0);
  matrix_t A2 = randu(n, m);

  display(A0);
  display(A1);
  display(A2);
  printf("SUM 1: %lf \n", sum(A1));
  printf("SUM 2: %lf \n", sum(A2));
  printf("MEAN 1: %lf \n", mean(A1));
  printf("MEAN 2: %lf \n", mean(A2));
  printf("MAX: %lf \n", max(A2));
  printf("MIN: %lf \n", min(A2));
  printf("LOG: ");
  display(log(A2));
  printf("\n  EXP: ");
  display(exp(A2));
  printf("\n  TANH: ");
  display(tanh(A2));
  printf("\n  POW 2: ");
  display(pow(A2, 2));
  printf("\n  TRANS: ");
  display(transpose(A2));

}


int main(int argc, char** argv) {

  //matrix_test();
  return 0;

}
