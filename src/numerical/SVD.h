#ifndef SVD_H
#define SVD_H

#include <stdlib.h>
#include <math.h>
#include "../core/Matrix.h"
#include "../core/vector.h"

#ifndef SVD_MAX_ITER
#  define SVD_MAX_ITER 100
#endif

#ifndef SVD_DEFAULT_TOL
#define SVD_DEFAULT_TOL 1e-10
#endif

typedef struct {
    Matrix *U;
    Vector *sigma;
    Matrix *Vt;
} SVDResult;

void svd_result_free(SVDResult *r);
int svd_raw(const double *A, int m, int n, double *U, double *sigma, double *Vt);
SVDResult* svd_decompose(const Matrix *A);
Vector* svd_solve(const SVDResult *svd, const Vector *b, double tol);
Matrix* svd_pinv(const SVDResult *svd, double tol);
int svd_rank(const SVDResult *svd, double tol);
double svd_condition_number(const SVDResult *svd, double tol);

#endif