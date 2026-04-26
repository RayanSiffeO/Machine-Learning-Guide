#include "LogisticRegression.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double logistic_bce_loss(const Matrix *X, const Vector *y, const Vector *w, Vector *grad_out) {
    if (!X || !y || !w) return -1.0;
    int n = X->rows;

    Vector *z = matrix_vec_mul(X, w);
    if (!z) return -1.0;

    Vector *y_hat = vector_sigmoid(z);
    vector_free(z);

    double loss = 0.0;
    double eps = 1e-12;
    for (int i = 0; i < n; i++) {
        double p = fmax(eps, fmin(1.0 - eps, y_hat->data[i]));
        loss += y->data[i] * log(p) + (1.0 - y->data[i]) * log(1.0 - p);
    }
    loss = -loss / n;

    if (grad_out) {
        Vector *error = vector_sub(y_hat, y);
        Matrix *XT = matrix_tras(X);
        Vector *grad_raw = matrix_vec_mul(XT, error);

        for (int j = 0; j < X->cols; j++) {
            grad_out->data[j] = grad_raw->data[j] / n;
        }

        vector_free(error);
        vector_free(grad_raw);
        matrix_free(XT);
    }

    vector_free(y_hat);
    return loss;
}

/* Single unified constructor — cfg is stored in the model and used at fit time. */
LogisticModel* logistic_regression_create(OptimizerConfig cfg, int epochs, int verbose) {
    LogisticModel *m = (LogisticModel *)calloc(1, sizeof(LogisticModel));
    if (!m) return NULL;
    m->config  = cfg;
    m->epochs  = epochs;
    m->verbose = verbose;
    m->opt     = NULL;
    m->weights = NULL;
    return m;
}

void logistic_regression_free(LogisticModel *m) {
    if (!m) return;
    vector_free(m->weights);
    optimizer_free(m->opt);
    free(m);
}

double logistic_regression_fit(LogisticModel *m, const Matrix *X, const Vector *y) {
    if (!m || !X || !y || X->rows != y->size) {
        fprintf(stderr, "logistic_regression_fit: argumentos invalidos\n");
        return -1.0;
    }

    int n_features = X->cols;

    if (!m->weights) {
        m->weights = vector_create(n_features);
        if (!m->weights) return -1.0;
    }

    if (!m->opt) {
        /* Build optimizer from the config stored at construction time. */
        m->opt = optimizer_create(m->config, n_features);
        if (!m->opt) return -1.0;
    }

    m->last_loss = gradient_descent(X, y, m->weights, m->opt, logistic_bce_loss, m->epochs, m->verbose);
    return m->last_loss;
}

Vector* logistic_regression_predict_proba(const LogisticModel *m, const Matrix *X) {
    if (!m || !m->weights || !X || X->cols != m->weights->size) {
        fprintf(stderr, "logistic_regression_predict_proba: argumentos invalidos\n");
        return NULL;
    }

    Vector *z     = matrix_vec_mul(X, m->weights);
    if (!z) return NULL;

    Vector *proba = vector_sigmoid(z);
    vector_free(z);
    return proba;
}

Vector* logistic_regression_predict(const LogisticModel *m, const Matrix *X) {
    Vector *proba = logistic_regression_predict_proba(m, X);
    if (!proba) return NULL;

    Vector *labels = vector_create(proba->size);
    if (!labels) { vector_free(proba); return NULL; }

    for (int i = 0; i < proba->size; i++)
        labels->data[i] = (proba->data[i] >= 0.5) ? 1.0 : 0.0;

    vector_free(proba);
    return labels;
}

double logistic_regression_score(const LogisticModel *m, const Matrix *X, const Vector *y) {
    if (!m || !X || !y) return -1.0;

    Vector *pred = logistic_regression_predict(m, X);
    if (!pred) return -1.0;

    int correct = 0;
    for (int i = 0; i < y->size; i++)
        if (pred->data[i] == y->data[i]) correct++;

    vector_free(pred);
    return (double)correct / y->size;
}
