#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vector.h"

typedef struct {
    int     rows;
    int     cols;
    double *data;   
} Matrix;

Matrix* matrix_create(int rows, int cols);
void    matrix_free(Matrix *m);
Matrix* matrix_copy(const Matrix *src);
Matrix* matrix_from_array(const double *arr, int rows, int cols);
void    matrix_fill(Matrix *m, double value);
void    matrix_zero(Matrix *m);
Matrix* matrix_identity(int n);
Matrix* matrix_diag(const Vector *d);
double  matrix_get(const Matrix *m, int i, int j);
void    matrix_set(Matrix *m, int i, int j, double value);
Vector* matrix_get_row(const Matrix *m, int i);
Vector* matrix_get_col(const Matrix *m, int j);
void    matrix_set_row(Matrix *m, int i, const Vector *v);
void    matrix_set_col(Matrix *m, int j, const Vector *v);
Matrix* matrix_tras(const Matrix *m);
#define matrix_transpose(m) matrix_tras(m) 
Matrix* matrix_reshape(const Matrix *m, int new_rows, int new_cols);
Matrix* matrix_slice(const Matrix *m, int r0, int r1, int c0, int c1);
Matrix* matrix_vstack(const Matrix *a, const Matrix *b);
Matrix* matrix_hstack(const Matrix *a, const Matrix *b);
Vector* matrix_flatten(const Matrix *m);
Matrix* matrix_add(const Matrix *a, const Matrix *b);
Matrix* matrix_sub(const Matrix *a, const Matrix *b);
Matrix* matrix_mul_elem(const Matrix *a, const Matrix *b);
Matrix* matrix_mul(const Matrix *A, const Matrix *B);
Vector* matrix_vec_mul(const Matrix *A, const Vector *x);
void    matrix_scale(Matrix *m, double scalar);
void    matrix_shift(Matrix *m, double scalar);
void    matrix_axpy(Matrix *A, double alpha, const Matrix *B);
Matrix* matrix_add_row_broadcast(const Matrix *m, const Vector *v);
Matrix* matrix_add_col_broadcast(const Matrix *m, const Vector *v);
double  matrix_sum(const Matrix *m);
double  matrix_mean(const Matrix *m);
double  matrix_frobenius_norm(const Matrix *m); 
double  matrix_max(const Matrix *m);
double  matrix_min(const Matrix *m);
double  matrix_trace(const Matrix *m); 
Vector* matrix_row_sum(const Matrix *m);
Vector* matrix_col_sum(const Matrix *m);
Vector* matrix_row_mean(const Matrix *m);
Vector* matrix_col_mean(const Matrix *m);
Vector* matrix_row_max(const Matrix *m);
Vector* matrix_col_max(const Matrix *m);
int*    matrix_lu(const Matrix *A, Matrix **L, Matrix **U);
Vector* matrix_solve(const Matrix *A, const Vector *b);
Matrix* matrix_inv(const Matrix *A);
double  matrix_det(const Matrix *A);
Matrix* matrix_cholesky(const Matrix *A);
Matrix* matrix_cov(const Matrix *X);
Matrix* matrix_standardise(const Matrix *X);
Matrix* matrix_minmax_norm(const Matrix *X);
Matrix* matrix_outer(const Vector *u, const Vector *v);
Matrix* matrix_softmax_rows(const Matrix *m);
Matrix* matrix_sigmoid(const Matrix *m);
Matrix* matrix_relu(const Matrix *m);
Matrix* matrix_tanh_elem(const Matrix *m);
Matrix* matrix_clip(const Matrix *m, double lo, double hi);
Matrix* matrix_log(const Matrix *m);
Matrix* matrix_exp(const Matrix *m);
int     matrix_check_dims(const Matrix *A, const Matrix *B, const char *fn);
void    matrix_print(const Matrix *m);
void    matrix_print_named(const char *name, const Matrix *m);
void    matrix_print_shape(const char *name, const Matrix *m);

#endif 