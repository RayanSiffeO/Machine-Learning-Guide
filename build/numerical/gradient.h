#ifndef GRADIENT_H
#define GRADIENT_H

#include "../../src/core/Matrix.h"
#include "../../src/core/vector.h"

typedef enum {
    OPTIM_SGD,
    OPTIM_MOMENTUM,
    OPTIM_RMSPROP,
    OPTIM_ADAM
} OptimizerType;

typedef struct {
    OptimizerType type;
    double lr;
    double momentum;
    double beta2;
    double epsilon;
    double decay;
} OptimizerConfig;

typedef struct {
    OptimizerConfig cfg;
    Vector *m;
    Vector *v;
    int     t;
} Optimizer;

OptimizerConfig optimizer_config_sgd      (double lr);
OptimizerConfig optimizer_config_momentum (double lr, double momentum);
OptimizerConfig optimizer_config_rmsprop  (double lr);
OptimizerConfig optimizer_config_adam     (double lr);
Optimizer* optimizer_create(OptimizerConfig cfg, int n);
void       optimizer_free(Optimizer *opt);
void       optimizer_reset(Optimizer *opt);
void       optimizer_step(Optimizer *opt, Vector *weights, const Vector *grad);

typedef double (*LossFn)(const Matrix *X, const Vector *y, const Vector *w, Vector *grad_out);
double gradient_descent(const Matrix *X, const Vector *y, Vector *w, Optimizer *opt,
                        LossFn loss_fn, int epochs, int verbose);

#endif