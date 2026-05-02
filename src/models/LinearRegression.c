#include "LinearRegression.h"
#include <stdio.h>
#include <stdlib.h>
 
LinearModel* linear_regression_create(double tol) {
    LinearModel *m = (LinearModel *)calloc(1, sizeof(LinearModel));
    if (!m) return NULL;
    m->tol = (tol > 0.0) ? tol : SVD_DEFAULT_TOL;
    m->r2  = 0.0;
    return m;
}
 
void linear_regression_free(LinearModel *m) {
    if (!m) return;
    vector_free(m->weights);
    free(m);
}
 
double linear_regression_fit(LinearModel *m, const Matrix *A, const Vector *b) {
    if (!m || !A || !b || A->rows != b->size) {
        fprintf(stderr, "linear_regression_fit: invalid arguments\n");
        return -1.0;
    }
 
    SVDResult *s = svd_decompose(A);
    if (!s) {
        fprintf(stderr, "linear_regression_fit: SVD failed\n");
        return -1.0;
    }
 
    vector_free(m->weights);
    m->weights = svd_solve(s, b, m->tol);
    svd_result_free(s);
 
    if (!m->weights) {
        fprintf(stderr, "linear_regression_fit: solve failed\n");
        return -1.0;
    }
 
    m->r2 = linear_regression_score(m, A, b);
    return m->r2;
}
 
Vector* linear_regression_predict(const LinearModel *m, const Matrix *X) {
    if (!m || !m->weights || !X || X->cols != m->weights->size) {
        fprintf(stderr, "linear_regression_predict: invalid arguments\n");
        return NULL;
    }
    return matrix_vec_mul(X, m->weights);
}
 
double linear_regression_score(const LinearModel *m,
                                const Matrix *X, const Vector *y) {
    if (!m || !y || !X) return -1.0;
 
    Vector *y_pred = linear_regression_predict(m, X);
    if (!y_pred) return -1.0;
 
    double y_mean = vector_mean(y);
    double ss_res = 0.0, ss_tot = 0.0;
 
    for (int i = 0; i < y->size; i++) {
        double r = y->data[i] - y_pred->data[i];
        double t = y->data[i] - y_mean;
        ss_res += r * r;
        ss_tot += t * t;
    }
 
    vector_free(y_pred);
    return (ss_tot > 0.0) ? 1.0 - ss_res / ss_tot : 1.0;
}