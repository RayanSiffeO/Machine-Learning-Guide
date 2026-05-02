#include "gradient.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

OptimizerConfig optimizer_config_sgd(double lr) {
    return (OptimizerConfig){
        .type     = OPTIM_SGD,
        .lr       = lr,
        .momentum = 0.0,
        .beta2    = 0.0,
        .epsilon  = 0.0,
        .decay    = 0.0
    };
}

OptimizerConfig optimizer_config_momentum(double lr, double momentum) {
    return (OptimizerConfig){
        .type     = OPTIM_MOMENTUM,
        .lr       = lr,
        .momentum = momentum,
        .beta2    = 0.0,
        .epsilon  = 1e-8,
        .decay    = 0.0
    };
}

OptimizerConfig optimizer_config_rmsprop(double lr) {
    return (OptimizerConfig){
        .type     = OPTIM_RMSPROP,
        .lr       = lr,
        .momentum = 0.0,
        .beta2    = 0.9,
        .epsilon  = 1e-8,
        .decay    = 0.0
    };
}

OptimizerConfig optimizer_config_adam(double lr) {
    return (OptimizerConfig){
        .type     = OPTIM_ADAM,
        .lr       = lr,
        .momentum = 0.9,
        .beta2    = 0.999,
        .epsilon  = 1e-8,
        .decay    = 0.0
    };
}

Optimizer* optimizer_create(OptimizerConfig cfg, int n) {
    Optimizer *opt = (Optimizer *)calloc(1, sizeof(Optimizer));
    if (!opt) return NULL;

    opt->cfg = cfg;
    opt->t   = 0;

    if (cfg.type == OPTIM_MOMENTUM ||
        cfg.type == OPTIM_RMSPROP  ||
        cfg.type == OPTIM_ADAM) {
        opt->m = vector_create(n);
        opt->v = vector_create(n);
        if (!opt->m || !opt->v) {
            optimizer_free(opt);
            return NULL;
        }
    }

    return opt;
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return;
    vector_free(opt->m);
    vector_free(opt->v);
    free(opt);
}

void optimizer_reset(Optimizer *opt) {
    if (!opt) return;
    opt->t = 0;
    if (opt->m) vector_zero(opt->m);
    if (opt->v) vector_zero(opt->v);
}

void optimizer_step(Optimizer *opt, Vector *w, const Vector *grad) {
    if (!opt || !w || !grad || w->size != grad->size) return;

    int    n   = w->size;
    double lr  = opt->cfg.lr;

    Vector *g = vector_copy(grad);
    if (!g) return;

    if (opt->cfg.decay > 0.0)
        vector_axpy(g, opt->cfg.decay, w);

    opt->t++;

    switch (opt->cfg.type) {

    case OPTIM_SGD:
        vector_axpy(w, -lr, g);
        break;

    case OPTIM_MOMENTUM:
        vector_scale(opt->m, opt->cfg.momentum);
        vector_axpy(opt->m, 1.0, g);
        vector_axpy(w, -lr, opt->m);
        break;

    case OPTIM_RMSPROP:
        for (int i = 0; i < n; i++) {
            opt->v->data[i] = opt->cfg.beta2 * opt->v->data[i]
                            + (1.0 - opt->cfg.beta2) * g->data[i] * g->data[i];
            w->data[i] -= lr * g->data[i]
                        / (sqrt(opt->v->data[i]) + opt->cfg.epsilon);
        }
        break;

    case OPTIM_ADAM: {
        double b1t = pow(opt->cfg.momentum, opt->t);
        double b2t = pow(opt->cfg.beta2,    opt->t);
        double lr_t = lr * sqrt(1.0 - b2t) / (1.0 - b1t);

        for (int i = 0; i < n; i++) {
            opt->m->data[i] = opt->cfg.momentum * opt->m->data[i]
                            + (1.0 - opt->cfg.momentum) * g->data[i];
            opt->v->data[i] = opt->cfg.beta2 * opt->v->data[i]
                            + (1.0 - opt->cfg.beta2) * g->data[i] * g->data[i];
            w->data[i] -= lr_t * opt->m->data[i]
                        / (sqrt(opt->v->data[i]) + opt->cfg.epsilon);
        }
        break;
    }
    }

    vector_free(g);
}

double gradient_descent(const Matrix *X, const Vector *y, Vector *w, Optimizer *opt, LossFn loss_fn, int epochs, int verbose) {
    if (!X || !y || !w || !opt || !loss_fn) return -1.0;

    Vector *grad = vector_create(w->size);
    if (!grad) return -1.0;

    double loss = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        vector_zero(grad);
        loss = loss_fn(X, y, w, grad);
        optimizer_step(opt, w, grad);

        if (verbose > 0 && (epoch + 1) % verbose == 0)
            printf("epoch %4d / %d   loss = %.6f\n", epoch + 1, epochs, loss);
    }

    vector_free(grad);
    return loss;
}