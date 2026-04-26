#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int     size;
    double *data;
} Vector;

Vector* vector_create(int size);
void    vector_free(Vector *v);
Vector* vector_copy(const Vector *src);
Vector* vector_from_array(const double *arr, int size);
void    vector_fill(Vector *v, double value);
void    vector_zero(Vector *v);
double  vector_get(const Vector *v, int i);
void    vector_set(Vector *v, int i, double value);
Vector* vector_add(const Vector *a, const Vector *b);
Vector* vector_sub(const Vector *a, const Vector *b);
Vector* vector_mul_elem(const Vector *a, const Vector *b);
Vector* vector_div_elem(const Vector *a, const Vector *b);
void    vector_scale(Vector *v, double scalar);
void    vector_shift(Vector *v, double scalar);
void    vector_axpy(Vector *v, double alpha, const Vector *u);
double  vector_dot(const Vector *a, const Vector *b);
double  vector_norm(const Vector *v);          
double  vector_norm1(const Vector *v);          
double  vector_norm_inf(const Vector *v);       
double  vector_sum(const Vector *v);
double  vector_mean(const Vector *v);
double  vector_variance(const Vector *v);     
double  vector_std(const Vector *v);            
double  vector_min(const Vector *v);
double  vector_max(const Vector *v);
int     vector_argmin(const Vector *v);
int     vector_argmax(const Vector *v);
void    vector_normalize(Vector *v);
Vector* vector_clip(const Vector *v, double lo, double hi);
Vector* vector_log(const Vector *v);
Vector* vector_exp(const Vector *v);
Vector* vector_sqrt(const Vector *v);
Vector* vector_pow(const Vector *v, double p);
Vector* vector_abs(const Vector *v);
Vector* vector_softmax(const Vector *v);
Vector* vector_sigmoid(const Vector *v);
Vector* vector_relu(const Vector *v);
Vector* vector_leaky_relu(const Vector *v, double alpha);
Vector* vector_tanh(const Vector *v);
Vector* vector_one_hot(int class_idx, int n_classes);
Vector* vector_slice(const Vector *v, int start, int end);
Vector* vector_concat(const Vector *a, const Vector *b);
void    vector_print(const Vector *v);
void    vector_print_named(const char *name, const Vector *v);
int     vector_check_dims(const Vector *a, const Vector *b, const char *fn);

#endif 