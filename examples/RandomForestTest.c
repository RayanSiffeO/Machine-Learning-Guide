#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
 
#include "build/Models/RandomForest.h"
#include "src/core/Matrix.h"
#include "src/core/vector.h"
 
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
    printf("[RandomForest] test_create_free ... ");
    RandomForest *rf = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 42);
    assert(rf != NULL);
    assert(rf->n_estimators == 10);
    assert(rf->bootstrap == 1);
    rf_free(rf);
    rf_free(NULL);
    printf("OK\n");
}
 
static void test_fit_metadata(Dataset ds) {
    printf("[RandomForest] test_fit_metadata ... ");
    RandomForest *rf = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 42);
    rf_fit(rf, ds.X, ds.y);
    assert(rf->n_trees_fitted == 10);
    assert(rf->n_features == ds.X->cols);
    assert(rf->n_classes >= 2);
    printf("OK  (n_classes=%d)\n", rf->n_classes);
    rf_free(rf);
}
 
static void test_predict_shape(Dataset ds) {
    printf("[RandomForest] test_predict_shape ... ");
    RandomForest *rf = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 42);
    rf_fit(rf, ds.X, ds.y);
    Vector *pred = rf_predict(rf, ds.X);
    assert(pred != NULL);
    assert(pred->size == ds.X->rows);
    for (int i = 0; i < pred->size; i++) {
        int cls = (int)round(pred->data[i]);
        assert(cls >= 0 && cls < rf->n_classes);
    }
    vector_free(pred);
    rf_free(rf);
    printf("OK\n");
}
 
static void test_predict_proba_valid(Dataset ds) {
    printf("[RandomForest] test_predict_proba_valid ... ");
    RandomForest *rf = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 42);
    rf_fit(rf, ds.X, ds.y);
    Matrix *proba = rf_predict_proba(rf, ds.X);
    assert(proba != NULL);
    assert(proba->rows == ds.X->rows);
    assert(proba->cols == rf->n_classes);
    for (int i = 0; i < proba->rows; i++) {
        double row_sum = 0.0;
        for (int c = 0; c < proba->cols; c++) {
            double v = proba->data[i * proba->cols + c];
            assert(v >= 0.0 && v <= 1.0);
            row_sum += v;
        }
        assert(fabs(row_sum - 1.0) < 1e-6);
    }
    matrix_free(proba);
    rf_free(rf);
    printf("OK\n");
}
 
static void test_feature_importances(Dataset ds) {
    printf("[RandomForest] test_feature_importances ... ");
    RandomForest *rf = rf_create(15, 5, 2, 1, -1, CRITERION_GINI, 42);
    rf_fit(rf, ds.X, ds.y);
    Vector *fi = rf_feature_importances(rf);
    assert(fi != NULL);
    assert(fi->size == ds.X->cols);
    double total = 0.0;
    for (int i = 0; i < fi->size; i++) {
        assert(fi->data[i] >= 0.0);
        total += fi->data[i];
    }
    assert(fabs(total - 1.0) < 1e-6);
    vector_free(fi);
    rf_free(rf);
    printf("OK\n");
}
 
static void test_score_range(Dataset ds) {
    printf("[RandomForest] test_score_range ... ");
    RandomForest *rf = rf_create(20, 6, 2, 1, -1, CRITERION_GINI, 42);
    rf_fit(rf, ds.X, ds.y);
    double score = rf_score(rf, ds.X, ds.y);
    assert(score >= 0.0 && score <= 1.0);
    rf_free(rf);
    printf("OK  (accuracy=%.4f)\n", score);
}
 
static void test_criteria(Dataset ds) {
    printf("[RandomForest] test_criteria (GINI/ENTROPY/MSE) ... ");
    Criterion criteria[] = {CRITERION_GINI, CRITERION_ENTROPY, CRITERION_MSE};
    for (int i = 0; i < 3; i++) {
        RandomForest *rf = rf_create(5, 4, 2, 1, -1, criteria[i], 42);
        rf_fit(rf, ds.X, ds.y);
        double s = rf_score(rf, ds.X, ds.y);
        assert(s >= 0.0 && s <= 1.0);
        rf_free(rf);
    }
    printf("OK\n");
}
 
static void test_reproducible(Dataset ds) {
    printf("[RandomForest] test_reproducible (misma semilla) ... ");
    RandomForest *rfa = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 999);
    RandomForest *rfb = rf_create(10, 5, 2, 1, -1, CRITERION_GINI, 999);
    rf_fit(rfa, ds.X, ds.y);
    rf_fit(rfb, ds.X, ds.y);
    Vector *pa = rf_predict(rfa, ds.X);
    Vector *pb = rf_predict(rfb, ds.X);
    assert(pa && pb && pa->size == pb->size);
    for (int i = 0; i < pa->size; i++)
        assert(pa->data[i] == pb->data[i]);
    vector_free(pa); vector_free(pb);
    rf_free(rfa); rf_free(rfb);
    printf("OK\n");
}
 
static void test_invalid_args(void) {
    printf("[RandomForest] test_invalid_args ... ");
    RandomForest *rf = rf_create(5, 3, 2, 1, -1, CRITERION_GINI, 1);
    rf_fit(rf, NULL, NULL);
    assert(rf->n_trees_fitted == 0);
    double Xd[] = {1.0, 2.0};
    Matrix *X = matrix_from_array(Xd, 1, 2);
    assert(rf_predict(rf, X) == NULL);
    matrix_free(X);
    rf_free(rf);
    rf_free(NULL);
    printf("OK\n");
}
 
// ── main ───────────────────────────────────────────────────────────────── 
 
int main(int argc, char **argv) {
    printf("\n=== RandomForest Tests ===\n");
 
    test_create_free();
    test_invalid_args();
 
    if (argc < 2) {
        printf("[AVISO] Sin CSV\n");
        printf("  Uso: %s datos.csv\n\n", argv[0]);
        return 0;
    }
 
    Dataset ds = load_csv(argv[1]);
    if (!ds.X || !ds.y) { fprintf(stderr, "Error cargando CSV\n"); return 1; }
 
    printf("Dataset: %d filas, %d features\n", ds.X->rows, ds.X->cols);
 
    test_fit_metadata(ds);
    test_predict_shape(ds);
    test_predict_proba_valid(ds);
    test_feature_importances(ds);
    test_score_range(ds);
    test_criteria(ds);
    test_reproducible(ds);
 
    matrix_free(ds.X);
    vector_free(ds.y);
 
    printf("=== Todos los tests pasaron ===\n\n");
    return 0;
}