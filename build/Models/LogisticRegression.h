#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "../../src/core/Matrix.h"
#include "../../src/core/vector.h"
#include "../numerical/gradient.h"

typedef struct {
    Vector         *weights;
    Optimizer      *opt;
    OptimizerConfig config;  
    int             epochs;
    int             verbose;
    double          last_loss;
} LogisticModel;

LogisticModel* logistic_regression_create(OptimizerConfig cfg, int epochs, int verbose);
void    logistic_regression_free(LogisticModel *m);
double  logistic_regression_fit(LogisticModel *m, const Matrix *X, const Vector *y);
Vector* logistic_regression_predict_proba(const LogisticModel *m, const Matrix *X);
Vector* logistic_regression_predict(const LogisticModel *m, const Matrix *X);
double  logistic_regression_score(const LogisticModel *m, const Matrix *X, const Vector *y);
double  logistic_bce_loss(const Matrix *X, const Vector *y, const Vector *w, Vector *grad_out);

#endif