#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
 
#include "../src/models/LogisticRegression.h"
#include "../src/core/Matrix.h"
#include "../src/core/vector.h"
 
// ── Lector de CSV ──────────────────────────────────────────────────────── 
 
typedef struct { Matrix *X; Vector *y; } Dataset;
 
static Dataset load_csv(const char *path) {
    Dataset ds = {NULL, NULL};
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "No se puede abrir: %s\n", path); return ds; }
 
    char line[4096];
    int rows = 0, cols = 0;
    while (fgets(line, sizeof(line), f)) {
        if (rows == 0) {
            char *p = line;
            while (*p) { if (*p++ == ',') cols++; }
            cols++;
        }
        rows++;
    }
    rewind(f);
 
    int n_features = cols - 1;
    double *Xbuf = malloc(rows * n_features * sizeof(double));
    double *ybuf = malloc(rows * sizeof(double));
    assert(Xbuf && ybuf);
 
    int r = 0;
    while (fgets(line, sizeof(line), f) && r < rows) {
        char *tok = strtok(line, ",\n");
        for (int c = 0; tok && c < cols; c++, tok = strtok(NULL, ",\n")) {
            double v = atof(tok);
            if (c < n_features) Xbuf[r * n_features + c] = v;
            else                ybuf[r] = v;
        }
        r++;
    }
    fclose(f);
 
    ds.X = matrix_from_array(Xbuf, rows, n_features);
    ds.y = vector_from_array(ybuf, rows);
    free(Xbuf); free(ybuf);
    return ds;
}
 
// ── Tests ──────────────────────────────────────────────────────────────── 
 
static void test_create_free(void) {
    printf("[LogisticRegression] test_create_free ... ");
    OptimizerConfig cfgs[] = {
        optimizer_config_sgd(0.1),
        optimizer_config_momentum(0.1, 0.9),
        optimizer_config_rmsprop(0.01),
        optimizer_config_adam(0.01),
    };
    for (int i = 0; i < 4; i++) {
        LogisticModel *m = logistic_regression_create(cfgs[i], 10, 0);
        assert(m != NULL);
        logistic_regression_free(m);
    }
    logistic_regression_free(NULL);
    printf("OK\n");
}
 
static void test_fit_loss_positive(Dataset ds) {
    printf("[LogisticRegression] test_fit_loss_positive ... ");
    OptimizerConfig cfg = optimizer_config_adam(0.01);
    LogisticModel *m = logistic_regression_create(cfg, 100, 0);
    double loss = logistic_regression_fit(m, ds.X, ds.y);
    assert(m->weights != NULL);
    assert(m->weights->size == ds.X->cols);
    assert(loss > 0.0);
    printf("OK  (loss=%.4f)\n", loss);
    logistic_regression_free(m);
}
 
static void test_loss_decreases_with_epochs(Dataset ds) {
    printf("[LogisticRegression] test_loss_decreases_with_epochs ... ");
    OptimizerConfig cfg = optimizer_config_adam(0.01);
    LogisticModel *m10  = logistic_regression_create(cfg, 10,  0);
    LogisticModel *m500 = logistic_regression_create(cfg, 500, 0);
    double l10  = logistic_regression_fit(m10,  ds.X, ds.y);
    double l500 = logistic_regression_fit(m500, ds.X, ds.y);
    assert(l500 < l10);
    printf("OK  (loss@10ep=%.4f > loss@500ep=%.4f)\n", l10, l500);
    logistic_regression_free(m10);
    logistic_regression_free(m500);
}
 
static void test_predict_proba_range(Dataset ds) {
    printf("[LogisticRegression] test_predict_proba_range ... ");
    OptimizerConfig cfg = optimizer_config_adam(0.01);
    LogisticModel *m = logistic_regression_create(cfg, 200, 0);
    logistic_regression_fit(m, ds.X, ds.y);
    Vector *proba = logistic_regression_predict_proba(m, ds.X);
    assert(proba != NULL);
    assert(proba->size == ds.X->rows);
    for (int i = 0; i < proba->size; i++)
        assert(proba->data[i] >= 0.0 && proba->data[i] <= 1.0);
    vector_free(proba);
    logistic_regression_free(m);
    printf("OK\n");
}
 
static void test_predict_binary(Dataset ds) {
    printf("[LogisticRegression] test_predict_binary ... ");
    OptimizerConfig cfg = optimizer_config_adam(0.01);
    LogisticModel *m = logistic_regression_create(cfg, 200, 0);
    logistic_regression_fit(m, ds.X, ds.y);
    Vector *pred = logistic_regression_predict(m, ds.X);
    assert(pred != NULL);
    assert(pred->size == ds.X->rows);
    for (int i = 0; i < pred->size; i++)
        assert(pred->data[i] == 0.0 || pred->data[i] == 1.0);
    vector_free(pred);
    logistic_regression_free(m);
    printf("OK\n");
}
 
static void test_score_range(Dataset ds) {
    printf("[LogisticRegression] test_score_range ... ");
    OptimizerConfig cfg = optimizer_config_adam(0.01);
    LogisticModel *m = logistic_regression_create(cfg, 300, 0);
    logistic_regression_fit(m, ds.X, ds.y);
    double acc = logistic_regression_score(m, ds.X, ds.y);
    assert(acc >= 0.0 && acc <= 1.0);
    logistic_regression_free(m);
    printf("OK  (accuracy=%.4f)\n", acc);
}
 
static void test_invalid_args(void) {
    printf("[LogisticRegression] test_invalid_args ... ");
    assert(logistic_regression_fit(NULL, NULL, NULL) < 0.0);
 
    double Xd[] = {1,2,3};  double yd[] = {0,1};
    Matrix *X = matrix_from_array(Xd, 3, 1);
    Vector *y = vector_from_array(yd, 2);
    OptimizerConfig cfg = optimizer_config_sgd(0.01);
    LogisticModel *m = logistic_regression_create(cfg, 10, 0);
    assert(logistic_regression_fit(m, X, y) < 0.0);
    assert(logistic_regression_predict_proba(m, X) == NULL);
    matrix_free(X); vector_free(y); logistic_regression_free(m);
    printf("OK\n");
}
 
// ── main ───────────────────────────────────────────────────────────────── 
 
int main(int argc, char **argv) {
    printf("\n=== LogisticRegression Tests ===\n");
 
    test_create_free();
    test_invalid_args();
 
    if (argc < 2) {
        printf("[AVISO] Sin CSV — omitiendo tests con datos.\n");
        printf("  Uso: %s datos.csv\n\n", argv[0]);
        return 0;
    }
 
    Dataset ds = load_csv(argv[1]);
    if (!ds.X || !ds.y) { fprintf(stderr, "Error cargando CSV\n"); return 1; }
 
    printf("Dataset: %d filas, %d features\n", ds.X->rows, ds.X->cols);
 
    test_fit_loss_positive(ds);
    test_loss_decreases_with_epochs(ds);
    test_predict_proba_range(ds);
    test_predict_binary(ds);
    test_score_range(ds);
 
    matrix_free(ds.X);
    vector_free(ds.y);
 
    printf("=== Todos los tests pasaron ===\n\n");
    return 0;
}