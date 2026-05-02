#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "build/Models/LinearRegression.h"
#include "src/core/Matrix.h"
#include "src/core/vector.h"

//── Lector de CSV ──────────────────────────────────────────────────────── 

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

// ── Test ──────────────────────────────────────────────────────────────── 

static void test_create_free(void) {
    printf("[LinearRegression] test_create_free ... ");
    LinearModel *m = linear_regression_create(1e-9);
    assert(m != NULL);
    assert(m->weights == NULL);
    linear_regression_free(m);
    linear_regression_free(NULL);
    printf("OK\n");
}

static void test_fit_returns_r2(Dataset ds) {
    printf("[LinearRegression] test_fit_returns_r2 ... ");
    LinearModel *m = linear_regression_create(1e-9);
    double r2 = linear_regression_fit(m, ds.X, ds.y);
    assert(m->weights != NULL);
    assert(m->weights->size == ds.X->cols);
    assert(r2 >= -1.0 && r2 <= 1.0 + 1e-6);
    printf("OK  (R²=%.4f)\n", r2);
    linear_regression_free(m);
}

static void test_predict_shape(Dataset ds) {
    printf("[LinearRegression] test_predict_shape ... ");
    LinearModel *m = linear_regression_create(1e-9);
    linear_regression_fit(m, ds.X, ds.y);
    Vector *pred = linear_regression_predict(m, ds.X);
    assert(pred != NULL);
    assert(pred->size == ds.X->rows);
    vector_free(pred);
    linear_regression_free(m);
    printf("OK\n");
}

static void test_score_consistency(Dataset ds) {
    printf("[LinearRegression] test_score_consistency ... ");
    LinearModel *m = linear_regression_create(1e-9);
    double r2_fit   = linear_regression_fit(m, ds.X, ds.y);
    double r2_score = linear_regression_score(m, ds.X, ds.y);
    assert(fabs(r2_fit - r2_score) < 1e-9);
    linear_regression_free(m);
    printf("OK  (R²_fit=%.4f == R²_score=%.4f)\n", r2_fit, r2_score);
}

static void test_invalid_args(void) {
    printf("[LinearRegression] test_invalid_args ... ");
    assert(linear_regression_fit(NULL, NULL, NULL) < 0.0);

    double Xd[] = {1,2,3};  double yd[] = {1,2};
    Matrix *X = matrix_from_array(Xd, 3, 1);
    Vector *y = vector_from_array(yd, 2);
    LinearModel *m = linear_regression_create(1e-9);
    assert(linear_regression_fit(m, X, y) < 0.0);
    assert(linear_regression_predict(m, X) == NULL);
    matrix_free(X); vector_free(y); linear_regression_free(m);
    printf("OK\n");
}

// ── main ───────────────────────────────────────────────────────────────── 

int main(int argc, char **argv) {
    printf("\n=== LinearRegression Tests ===\n");

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

    test_fit_returns_r2(ds);
    test_predict_shape(ds);
    test_score_consistency(ds);

    matrix_free(ds.X);
    vector_free(ds.y);

    printf("=== Todos los tests pasaron ===\n\n");
    return 0;
}