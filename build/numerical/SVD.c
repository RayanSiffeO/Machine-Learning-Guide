#include "SVD.h"
#include <string.h>
#include <stdio.h>

static double nr_sign(double a, double b) { return b >= 0.0 ?  fabs(a) : -fabs(a); }
static double nr_sqr (double x)           { return x * x; }
static double nr_max2(double a, double b) { return a > b ? a : b; }

static double pythag(double a, double b) {
    double absa = fabs(a), absb = fabs(b);
    if (absa > absb)
        return absa * sqrt(1.0 + nr_sqr(absb / absa));
    return (absb == 0.0) ? 0.0 : absb * sqrt(1.0 + nr_sqr(absa / absb));
}

#define A2(ptr, cols, i, j)  ((ptr)[(i)*(cols)+(j)])

static int svdcmp(double *a, int m, int n, double *w, double *v) {
    int    flag, i, its, j, jj, k, l, nm;
    double anorm, c, f, g, h, s, scale, x, y, z;

    double *rv1 = (double *)malloc(n * sizeof(double));
    if (!rv1) return -1;

    l = nm = 0;
    g = scale = anorm = 0.0;

    for (i = 0; i < n; i++) {
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m) {
            for (k = i; k < m; k++) scale += fabs(A2(a, n, k, i));
            if (scale != 0.0) {
                for (k = i; k < m; k++) {
                    A2(a, n, k, i) /= scale;
                    s += A2(a, n, k, i) * A2(a, n, k, i);
                }
                f = A2(a, n, i, i);
                g = -nr_sign(sqrt(s), f);
                h = f * g - s;
                A2(a, n, i, i) = f - g;
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = i; k < m; k++)
                        s += A2(a, n, k, i) * A2(a, n, k, j);
                    f = s / h;
                    for (k = i; k < m; k++)
                        A2(a, n, k, j) += f * A2(a, n, k, i);
                }
                for (k = i; k < m; k++) A2(a, n, k, i) *= scale;
            }
        }
        w[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) scale += fabs(A2(a, n, i, k));
            if (scale != 0.0) {
                for (k = l; k < n; k++) {
                    A2(a, n, i, k) /= scale;
                    s += A2(a, n, i, k) * A2(a, n, i, k);
                }
                f = A2(a, n, i, l);
                g = -nr_sign(sqrt(s), f);
                h = f * g - s;
                A2(a, n, i, l) = f - g;
                for (k = l; k < n; k++) rv1[k] = A2(a, n, i, k) / h;
                for (j = l; j < m; j++) {
                    for (s = 0.0, k = l; k < n; k++)
                        s += A2(a, n, j, k) * A2(a, n, i, k);
                    for (k = l; k < n; k++)
                        A2(a, n, j, k) += s * rv1[k];
                }
                for (k = l; k < n; k++) A2(a, n, i, k) *= scale;
            }
        }
        anorm = nr_max2(anorm, fabs(w[i]) + fabs(rv1[i]));
    }

    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g != 0.0) {
                for (j = l; j < n; j++)
                    A2(v, n, j, i) = (A2(a, n, i, j) / A2(a, n, i, l)) / g;
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < n; k++)
                        s += A2(a, n, i, k) * A2(v, n, k, j);
                    for (k = l; k < n; k++)
                        A2(v, n, k, j) += s * A2(v, n, k, i);
                }
            }
            for (j = l; j < n; j++)
                A2(v, n, i, j) = A2(v, n, j, i) = 0.0;
        }
        A2(v, n, i, i) = 1.0;
        g = rv1[i];
        l = i;
    }

    for (i = (m < n ? m : n) - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        for (j = l; j < n; j++) A2(a, n, i, j) = 0.0;
        if (g != 0.0) {
            g = 1.0 / g;
            for (j = l; j < n; j++) {
                for (s = 0.0, k = l; k < m; k++)
                    s += A2(a, n, k, i) * A2(a, n, k, j);
                f = (s / A2(a, n, i, i)) * g;
                for (k = i; k < m; k++)
                    A2(a, n, k, j) += f * A2(a, n, k, i);
            }
            for (j = i; j < m; j++) A2(a, n, j, i) *= g;
        } else {
            for (j = i; j < m; j++) A2(a, n, j, i) = 0.0;
        }
        A2(a, n, i, i) += 1.0;
    }

    for (k = n - 1; k >= 0; k--) {
        for (its = 0; its < SVD_MAX_ITER; its++) {
            flag = 1;
            for (l = k; l >= 0; l--) {
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm) { flag = 0; break; }
                if (fabs(w[nm])  + anorm == anorm) break;
            }
            if (flag) {
                c = 0.0; s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    if (fabs(f) + anorm == anorm) break;
                    g = w[i]; h = pythag(f, g); w[i] = h;
                    h = 1.0 / h; c = g * h; s = -f * h;
                    for (j = 0; j < m; j++) {
                        y = A2(a, n, j, nm); z = A2(a, n, j, i);
                        A2(a, n, j, nm) = y * c + z * s;
                        A2(a, n, j, i)  = z * c - y * s;
                    }
                }
            }
            z = w[k];
            if (l == k) {
                if (z < 0.0) {
                    w[k] = -z;
                    for (j = 0; j < n; j++) A2(v, n, j, k) = -A2(v, n, j, k);
                }
                break;
            }
            if (its == SVD_MAX_ITER - 1) { free(rv1); return -1; }
            x = w[l]; nm = k - 1; y = w[nm]; g = rv1[nm]; h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * (y / (f + nr_sign(g, f)) - h)) / x;
            c = s = 1.0;
            for (j = l; j <= nm; j++) {
                i = j + 1; g = rv1[i]; y = w[i]; h = s * g; g = c * g;
                z = pythag(f, h); rv1[j] = z;
                c = f / z; s = h / z;
                f = x * c + g * s; g = g * c - x * s; h = y * s; y *= c;
                for (jj = 0; jj < n; jj++) {
                    x = A2(v, n, jj, j); z = A2(v, n, jj, i);
                    A2(v, n, jj, j) = x * c + z * s;
                    A2(v, n, jj, i) = z * c - x * s;
                }
                z = pythag(f, h); w[j] = z;
                if (z != 0.0) { z = 1.0 / z; c = f * z; s = h * z; }
                f = c * g + s * y; x = c * y - s * g;
                for (jj = 0; jj < m; jj++) {
                    y = A2(a, n, jj, j); z = A2(a, n, jj, i);
                    A2(a, n, jj, j) = y * c + z * s;
                    A2(a, n, jj, i) = z * c - y * s;
                }
            }
            rv1[l] = 0.0; rv1[k] = f; w[k] = x;
        }
    }

    free(rv1);
    return 0;
}

int svd_raw(const double *A, int m, int n,
            double *U, double *sigma, double *Vt) {
    double *a = (double *)malloc(m * n * sizeof(double));
    double *v = (double *)malloc(n * n * sizeof(double));
    if (!a || !v) { free(a); free(v); return -1; }

    memcpy(a, A, m * n * sizeof(double));

    if (svdcmp(a, m, n, sigma, v) != 0) {
        free(a); free(v); return -1;
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (fabs(sigma[j]) > fabs(sigma[i])) {
                double tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
                for (int k = 0; k < m; k++) {
                    tmp = A2(a, n, k, i); A2(a, n, k, i) = A2(a, n, k, j); A2(a, n, k, j) = tmp;
                }
                for (int k = 0; k < n; k++) {
                    tmp = A2(v, n, k, i); A2(v, n, k, i) = A2(v, n, k, j); A2(v, n, k, j) = tmp;
                }
            }
        }
    }

    memcpy(U, a, m * n * sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A2(Vt, n, i, j) = A2(v, n, j, i);

    free(a);
    free(v);
    return 0;
}

void svd_result_free(SVDResult *r) {
    if (!r) return;
    matrix_free(r->U);
    vector_free(r->sigma);
    matrix_free(r->Vt);
    free(r);
}

SVDResult* svd_decompose(const Matrix *A) {
    if (!A) return NULL;
    int m = A->rows, n = A->cols;

    SVDResult *r = (SVDResult *)calloc(1, sizeof(SVDResult));
    if (!r) return NULL;

    r->U     = matrix_create(m, n);
    r->sigma = vector_create(n);
    r->Vt    = matrix_create(n, n);

    if (!r->U || !r->sigma || !r->Vt) {
        svd_result_free(r);
        return NULL;
    }

    if (svd_raw(A->data, m, n,
                r->U->data, r->sigma->data, r->Vt->data) != 0) {
        svd_result_free(r);
        return NULL;
    }

    return r;
}

Vector* svd_solve(const SVDResult *s, const Vector *b, double tol) {
    if (!s || !b) return NULL;

    int m = s->U->rows;
    int n = s->U->cols;

    if (b->size != m) {
        fprintf(stderr, "svd_solve: b->size (%d) != U->rows (%d)\n", b->size, m);
        return NULL;
    }

    if (tol <= 0.0) tol = SVD_DEFAULT_TOL;

    Vector *tmp = vector_create(n);
    if (!tmp) return NULL;

    for (int j = 0; j < n; j++) {
        double dot = 0.0;
        for (int i = 0; i < m; i++)
            dot += s->U->data[i * n + j] * b->data[i];
        tmp->data[j] = (s->sigma->data[j] > tol) ? dot / s->sigma->data[j] : 0.0;
    }

    Vector *w = vector_create(n);
    if (!w) { vector_free(tmp); return NULL; }

    for (int j = 0; j < n; j++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++)
            acc += s->Vt->data[k * n + j] * tmp->data[k];
        w->data[j] = acc;
    }

    vector_free(tmp);
    return w;
}

Matrix* svd_pinv(const SVDResult *s, double tol) {
    if (!s) return NULL;

    int m = s->U->rows;
    int n = s->U->cols;

    if (tol <= 0.0) tol = SVD_DEFAULT_TOL;

    Matrix *pinv = matrix_create(n, m);
    if (!pinv) return NULL;

    for (int i = 0; i < m; i++) {

        for (int k = 0; k < n; k++) {
            double sigma_k = s->sigma->data[k];
            double u_ik    = s->U->data[i * n + k];
            double scaled  = (sigma_k > tol) ? u_ik / sigma_k : 0.0;

            for (int j = 0; j < n; j++)
                pinv->data[j * m + i] += s->Vt->data[k * n + j] * scaled;
        }
    }

    return pinv;
}

int svd_rank(const SVDResult *s, double tol) {
    if (!s) return 0;
    if (tol <= 0.0) tol = SVD_DEFAULT_TOL;
    int r = 0;
    for (int i = 0; i < s->sigma->size; i++)
        if (s->sigma->data[i] > tol) r++;
    return r;
}

double svd_condition_number(const SVDResult *s, double tol) {
    if (!s || s->sigma->size == 0) return NAN;
    if (tol <= 0.0) tol = SVD_DEFAULT_TOL;

    double s_max = s->sigma->data[0];
    double s_min = s_max;
    for (int i = 1; i < s->sigma->size; i++) {
        double v = s->sigma->data[i];
        if (v > tol && v < s_min) s_min = v;
    }
    return (s_min > tol) ? s_max / s_min : INFINITY;
}