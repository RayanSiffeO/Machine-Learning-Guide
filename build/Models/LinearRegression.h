#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "../../src/core/Matrix.h"
#include "../../src/core/vector.h"
#include "../numerical/SVD.h"

typedef struct {
    Vector *weights;  
    double  tol;     
    double  r2;        
} LinearModel;

LinearModel* linear_regression_create(double tol);
void         linear_regression_free(LinearModel *m);
double       linear_regression_fit(LinearModel *m, const Matrix *A, const Vector *b);
Vector*      linear_regression_predict(const LinearModel *m, const Matrix *X);
double       linear_regression_score(const LinearModel *m, const Matrix *X, const Vector *y);

#endif