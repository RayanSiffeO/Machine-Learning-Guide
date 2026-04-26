#include "Matrix.h"

int matrix_check_dims(const Matrix *A, const Matrix *B, const char *fn) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "%s: shape mismatch (%dx%d vs %dx%d)\n", fn,
                A ? A->rows : -1, A ? A->cols : -1,
                B ? B->rows : -1, B ? B->cols : -1);
        return 0;
    }
    return 1;
}

Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;

    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    if (!m->data) { free(m); return NULL; }

    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_copy(const Matrix *src) {
    if (!src) return NULL;

    Matrix *dst = matrix_create(src->rows, src->cols);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
    return dst;
}

Matrix* matrix_from_array(const double *arr, int rows, int cols) {
    if (!arr || rows <= 0 || cols <= 0) return NULL;

    Matrix *m = matrix_create(rows, cols);
    if (!m) return NULL;

    memcpy(m->data, arr, rows * cols * sizeof(double));
    return m;
}

void matrix_fill(Matrix *m, double value) {
    if (!m) return;
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) m->data[i] = value;
}

void matrix_zero(Matrix *m) {
    if (m) memset(m->data, 0, m->rows * m->cols * sizeof(double));
}

Matrix* matrix_identity(int n) {
    if (n <= 0) return NULL;

    Matrix *I = matrix_create(n, n);
    if (!I) return NULL;

    for (int i = 0; i < n; i++)
        I->data[i * n + i] = 1.0;

    return I;
}

Matrix* matrix_diag(const Vector *d) {
    if (!d) return NULL;
    int n = d->size;
    Matrix *D = matrix_create(n, n);
    if (!D) return NULL;
    for (int i = 0; i < n; i++)
        D->data[i * n + i] = d->data[i];
    return D;
}

double matrix_get(const Matrix *m, int i, int j) {
    if (!m || i < 0 || i >= m->rows || j < 0 || j >= m->cols) {
        fprintf(stderr, "matrix_get: index (%d,%d) out of range for %dx%d\n",
                i, j, m ? m->rows : -1, m ? m->cols : -1);
        return NAN;
    }
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, int i, int j, double value) {
    if (!m || i < 0 || i >= m->rows || j < 0 || j >= m->cols) {
        fprintf(stderr, "matrix_set: index (%d,%d) out of range for %dx%d\n",
                i, j, m ? m->rows : -1, m ? m->cols : -1);
        return;
    }
    m->data[i * m->cols + j] = value;
}

Vector* matrix_get_row(const Matrix *m, int i) {
    if (!m || i < 0 || i >= m->rows) return NULL;
    return vector_from_array(m->data + i * m->cols, m->cols);
}

Vector* matrix_get_col(const Matrix *m, int j) {
    if (!m || j < 0 || j >= m->cols) return NULL;
    Vector *v = vector_create(m->rows);
    if (!v) return NULL;
    for (int i = 0; i < m->rows; i++)
        v->data[i] = m->data[i * m->cols + j];
    return v;
}

void matrix_set_row(Matrix *m, int i, const Vector *v) {
    if (!m || !v || i < 0 || i >= m->rows || v->size != m->cols) {
        fprintf(stderr, "matrix_set_row: invalid arguments\n");
        return;
    }
    memcpy(m->data + i * m->cols, v->data, m->cols * sizeof(double));
}

void matrix_set_col(Matrix *m, int j, const Vector *v) {
    if (!m || !v || j < 0 || j >= m->cols || v->size != m->rows) {
        fprintf(stderr, "matrix_set_col: invalid arguments\n");
        return;
    }
    for (int i = 0; i < m->rows; i++)
        m->data[i * m->cols + j] = v->data[i];
}

Matrix* matrix_tras(const Matrix *m) {
    if (!m) return NULL;

    Matrix *t = matrix_create(m->cols, m->rows);
    if (!t) return NULL;

    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            t->data[j * m->rows + i] = m->data[i * m->cols + j];

    return t;
}

Matrix* matrix_reshape(const Matrix *m, int new_rows, int new_cols) {
    if (!m || new_rows <= 0 || new_cols <= 0) return NULL;
    if (new_rows * new_cols != m->rows * m->cols) {
        fprintf(stderr, "matrix_reshape: %dx%d → %dx%d element count mismatch\n",
                m->rows, m->cols, new_rows, new_cols);
        return NULL;
    }
    Matrix *r = matrix_create(new_rows, new_cols);
    if (!r) return NULL;
    memcpy(r->data, m->data, m->rows * m->cols * sizeof(double));
    return r;
}

Matrix* matrix_slice(const Matrix *m, int r0, int r1, int c0, int c1) {
    if (!m || r0 < 0 || r1 > m->rows || c0 < 0 || c1 > m->cols ||
        r0 >= r1 || c0 >= c1) {
        fprintf(stderr, "matrix_slice: invalid range [%d:%d, %d:%d] for %dx%d\n",
                r0, r1, c0, c1, m ? m->rows : -1, m ? m->cols : -1);
        return NULL;
    }
    int nr = r1 - r0, nc = c1 - c0;
    Matrix *s = matrix_create(nr, nc);
    if (!s) return NULL;
    for (int i = 0; i < nr; i++)
        memcpy(s->data + i * nc,
               m->data + (r0 + i) * m->cols + c0,
               nc * sizeof(double));
    return s;
}

Matrix* matrix_vstack(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->cols != b->cols) {
        fprintf(stderr, "matrix_vstack: cols must match (%d vs %d)\n",
                a ? a->cols : -1, b ? b->cols : -1);
        return NULL;
    }
    Matrix *c = matrix_create(a->rows + b->rows, a->cols);
    if (!c) return NULL;
    memcpy(c->data,
           a->data, a->rows * a->cols * sizeof(double));
    memcpy(c->data + a->rows * a->cols,
           b->data, b->rows * b->cols * sizeof(double));
    return c;
}

Matrix* matrix_hstack(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows) {
        fprintf(stderr, "matrix_hstack: rows must match (%d vs %d)\n",
                a ? a->rows : -1, b ? b->rows : -1);
        return NULL;
    }
    int rows = a->rows;
    int ca = a->cols, cb = b->cols, cc = ca + cb;
    Matrix *c = matrix_create(rows, cc);
    if (!c) return NULL;
    for (int i = 0; i < rows; i++) {
        memcpy(c->data + i * cc,       a->data + i * ca, ca * sizeof(double));
        memcpy(c->data + i * cc + ca,  b->data + i * cb, cb * sizeof(double));
    }
    return c;
}

Vector* matrix_flatten(const Matrix *m) {
    if (!m) return NULL;
    return vector_from_array(m->data, m->rows * m->cols);
}

Matrix* matrix_add(const Matrix *a, const Matrix *b) {
    if (!matrix_check_dims(a, b, "matrix_add")) return NULL;

    Matrix *c = matrix_create(a->rows, a->cols);
    if (!c) return NULL;
    int total = a->rows * a->cols;
    for (int i = 0; i < total; i++) c->data[i] = a->data[i] + b->data[i];
    return c;
}

Matrix* matrix_sub(const Matrix *a, const Matrix *b) {
    if (!matrix_check_dims(a, b, "matrix_sub")) return NULL;

    Matrix *c = matrix_create(a->rows, a->cols);
    if (!c) return NULL;
    int total = a->rows * a->cols;
    for (int i = 0; i < total; i++) c->data[i] = a->data[i] - b->data[i];
    return c;
}

Matrix* matrix_mul_elem(const Matrix *a, const Matrix *b) {
    if (!matrix_check_dims(a, b, "matrix_mul_elem")) return NULL;

    Matrix *c = matrix_create(a->rows, a->cols);
    if (!c) return NULL;
    int total = a->rows * a->cols;
    for (int i = 0; i < total; i++) c->data[i] = a->data[i] * b->data[i];
    return c;
}

Matrix* matrix_mul(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->cols != B->rows) {
        fprintf(stderr, "matrix_mul: incompatible shapes (%dx%d) x (%dx%d)\n",
                A ? A->rows : -1, A ? A->cols : -1,
                B ? B->rows : -1, B ? B->cols : -1);
        return NULL;
    }
    Matrix *C = matrix_create(A->rows, B->cols);
    if (!C) return NULL;

    /* Loop order i-k-j exploits row-major cache for A. */
    for (int i = 0; i < A->rows; i++)
        for (int k = 0; k < A->cols; k++) {
            double a_ik = A->data[i * A->cols + k];
            for (int j = 0; j < B->cols; j++)
                C->data[i * C->cols + j] += a_ik * B->data[k * B->cols + j];
        }
    return C;
}

Vector* matrix_vec_mul(const Matrix *A, const Vector *x) {
    if (!A || !x || A->cols != x->size) {
        fprintf(stderr, "matrix_vec_mul: shape mismatch (%dx%d) x (%d)\n",
                A ? A->rows : -1, A ? A->cols : -1, x ? x->size : -1);
        return NULL;
    }
    Vector *y = vector_create(A->rows);
    if (!y) return NULL;
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int j = 0; j < A->cols; j++)
            acc += A->data[i * A->cols + j] * x->data[j];
        y->data[i] = acc;
    }
    return y;
}

void matrix_scale(Matrix *m, double scalar) {
    if (!m) return;
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) m->data[i] *= scalar;
}

void matrix_shift(Matrix *m, double scalar) {
    if (!m) return;
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) m->data[i] += scalar;
}

void matrix_axpy(Matrix *A, double alpha, const Matrix *B) {
    if (!matrix_check_dims(A, B, "matrix_axpy")) return;
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) A->data[i] += alpha * B->data[i];
}

Matrix* matrix_add_row_broadcast(const Matrix *m, const Vector *v) {
    if (!m || !v || v->size != m->cols) {
        fprintf(stderr, "matrix_add_row_broadcast: v->size (%d) != cols (%d)\n",
                v ? v->size : -1, m ? m->cols : -1);
        return NULL;
    }
    Matrix *c = matrix_copy(m);
    if (!c) return NULL;
    for (int i = 0; i < c->rows; i++)
        for (int j = 0; j < c->cols; j++)
            c->data[i * c->cols + j] += v->data[j];
    return c;
}

Matrix* matrix_add_col_broadcast(const Matrix *m, const Vector *v) {
    if (!m || !v || v->size != m->rows) {
        fprintf(stderr, "matrix_add_col_broadcast: v->size (%d) != rows (%d)\n",
                v ? v->size : -1, m ? m->rows : -1);
        return NULL;
    }
    Matrix *c = matrix_copy(m);
    if (!c) return NULL;
    for (int i = 0; i < c->rows; i++)
        for (int j = 0; j < c->cols; j++)
            c->data[i * c->cols + j] += v->data[i];
    return c;
}

double matrix_sum(const Matrix *m) {
    if (!m) return NAN;
    double acc = 0.0;
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) acc += m->data[i];
    return acc;
}

double matrix_mean(const Matrix *m) {
    if (!m || m->rows * m->cols == 0) return NAN;
    return matrix_sum(m) / (m->rows * m->cols);
}

double matrix_frobenius_norm(const Matrix *m) {
    if (!m) return NAN;
    double acc = 0.0;
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) acc += m->data[i] * m->data[i];
    return sqrt(acc);
}

double matrix_max(const Matrix *m) {
    if (!m || m->rows * m->cols == 0) return NAN;
    double mx = m->data[0];
    int total = m->rows * m->cols;
    for (int i = 1; i < total; i++)
        if (m->data[i] > mx) mx = m->data[i];
    return mx;
}

double matrix_min(const Matrix *m) {
    if (!m || m->rows * m->cols == 0) return NAN;
    double mn = m->data[0];
    int total = m->rows * m->cols;
    for (int i = 1; i < total; i++)
        if (m->data[i] < mn) mn = m->data[i];
    return mn;
}

double matrix_trace(const Matrix *m) {
    if (!m || m->rows != m->cols) {
        fprintf(stderr, "matrix_trace: matrix must be square\n");
        return NAN;
    }
    double acc = 0.0;
    for (int i = 0; i < m->rows; i++) acc += m->data[i * m->cols + i];
    return acc;
}

Vector* matrix_row_sum(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = vector_create(m->rows);
    if (!v) return NULL;
    for (int i = 0; i < m->rows; i++) {
        double acc = 0.0;
        for (int j = 0; j < m->cols; j++) acc += m->data[i * m->cols + j];
        v->data[i] = acc;
    }
    return v;
}

Vector* matrix_col_sum(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = vector_create(m->cols);
    if (!v) return NULL;
    for (int j = 0; j < m->cols; j++) {
        double acc = 0.0;
        for (int i = 0; i < m->rows; i++) acc += m->data[i * m->cols + j];
        v->data[j] = acc;
    }
    return v;
}

Vector* matrix_row_mean(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = matrix_row_sum(m);
    if (!v) return NULL;
    vector_scale(v, 1.0 / m->cols);
    return v;
}

Vector* matrix_col_mean(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = matrix_col_sum(m);
    if (!v) return NULL;
    vector_scale(v, 1.0 / m->rows);
    return v;
}

Vector* matrix_row_max(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = vector_create(m->rows);
    if (!v) return NULL;
    for (int i = 0; i < m->rows; i++) {
        double mx = m->data[i * m->cols];
        for (int j = 1; j < m->cols; j++) {
            double val = m->data[i * m->cols + j];
            if (val > mx) mx = val;
        }
        v->data[i] = mx;
    }
    return v;
}

Vector* matrix_col_max(const Matrix *m) {
    if (!m) return NULL;
    Vector *v = vector_create(m->cols);
    if (!v) return NULL;
    for (int j = 0; j < m->cols; j++) {
        double mx = m->data[j];
        for (int i = 1; i < m->rows; i++) {
            double val = m->data[i * m->cols + j];
            if (val > mx) mx = val;
        }
        v->data[j] = mx;
    }
    return v;
}

int* matrix_lu(const Matrix *A, Matrix **L_out, Matrix **U_out) {
    if (!A || A->rows != A->cols) {
        fprintf(stderr, "matrix_lu: requires square matrix\n");
        return NULL;
    }
    int n = A->rows;
    Matrix *U = matrix_copy(A);
    Matrix *L = matrix_identity(n);
    int *piv  = (int*)malloc(n * sizeof(int));
    if (!U || !L || !piv) {
        matrix_free(U); matrix_free(L); free(piv);
        return NULL;
    }
    for (int i = 0; i < n; i++) piv[i] = i;

    for (int k = 0; k < n; k++) {
        /* Find pivot */
        int p = k;
        double maxval = fabs(U->data[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = fabs(U->data[i * n + k]);
            if (v > maxval) { maxval = v; p = i; }
        }
        if (p != k) {
            int tmp = piv[k]; piv[k] = piv[p]; piv[p] = tmp;
            for (int j = 0; j < n; j++) {
                double t = U->data[k * n + j];
                U->data[k * n + j] = U->data[p * n + j];
                U->data[p * n + j] = t;
            }
            for (int j = 0; j < k; j++) {
                double t = L->data[k * n + j];
                L->data[k * n + j] = L->data[p * n + j];
                L->data[p * n + j] = t;
            }
        }
        double pivot = U->data[k * n + k];
        if (fabs(pivot) < 1e-14) continue; 

        for (int i = k + 1; i < n; i++) {
            double factor = U->data[i * n + k] / pivot;
            L->data[i * n + k] = factor;
            for (int j = k; j < n; j++)
                U->data[i * n + j] -= factor * U->data[k * n + j];
        }
    }
    *L_out = L;
    *U_out = U;
    return piv;
}

static Vector* forward_sub(const Matrix *L, const Vector *b) {
    int n = L->rows;
    Vector *y = vector_create(n);
    if (!y) return NULL;
    for (int i = 0; i < n; i++) {
        double acc = b->data[i];
        for (int j = 0; j < i; j++)
            acc -= L->data[i * n + j] * y->data[j];
        y->data[i] = acc; 
    }
    return y;
}

static Vector* back_sub(const Matrix *U, const Vector *y) {
    int n = U->rows;
    Vector *x = vector_create(n);
    if (!x) return NULL;
    for (int i = n - 1; i >= 0; i--) {
        double acc = y->data[i];
        for (int j = i + 1; j < n; j++)
            acc -= U->data[i * n + j] * x->data[j];
        double diag = U->data[i * n + i];
        if (fabs(diag) < 1e-14) {
            fprintf(stderr, "back_sub: zero diagonal at %d (singular)\n", i);
            vector_free(x);
            return NULL;
        }
        x->data[i] = acc / diag;
    }
    return x;
}

Vector* matrix_solve(const Matrix *A, const Vector *b) {
    if (!A || !b || A->rows != A->cols || A->rows != b->size) {
        fprintf(stderr, "matrix_solve: invalid shapes\n");
        return NULL;
    }
    int n = A->rows;
    Matrix *L = NULL, *U = NULL;
    int *piv = matrix_lu(A, &L, &U);
    if (!piv) return NULL;

    Vector *pb = vector_create(n);
    if (!pb) { matrix_free(L); matrix_free(U); free(piv); return NULL; }
    for (int i = 0; i < n; i++) pb->data[i] = b->data[piv[i]];

    Vector *y = forward_sub(L, pb);
    vector_free(pb);
    matrix_free(L);

    Vector *x = back_sub(U, y);
    vector_free(y);
    matrix_free(U);
    free(piv);
    return x;
}

Matrix* matrix_inv(const Matrix *A) {
    if (!A || A->rows != A->cols) {
        fprintf(stderr, "matrix_inv: requires square matrix\n");
        return NULL;
    }
    int n = A->rows;
    Matrix *Inv = matrix_create(n, n);
    if (!Inv) return NULL;

    for (int j = 0; j < n; j++) {

        Vector *ej = vector_create(n);
        if (!ej) { matrix_free(Inv); return NULL; }
        ej->data[j] = 1.0;

        Vector *col = matrix_solve(A, ej);
        vector_free(ej);
        if (!col) { matrix_free(Inv); return NULL; }

        for (int i = 0; i < n; i++)
            Inv->data[i * n + j] = col->data[i];
        vector_free(col);
    }
    return Inv;
}

double matrix_det(const Matrix *A) {
    if (!A || A->rows != A->cols) {
        fprintf(stderr, "matrix_det: requires square matrix\n");
        return NAN;
    }
    int n = A->rows;
    Matrix *L = NULL, *U = NULL;
    int *piv = matrix_lu(A, &L, &U);
    if (!piv) return NAN;


    double d = 1.0;
    for (int i = 0; i < n; i++) d *= U->data[i * n + i];

    /* Count swaps */
    int *visited = (int*)calloc(n, sizeof(int));
    int swaps = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            int cycle = 0, j = i;
            while (!visited[j]) { visited[j] = 1; j = piv[j]; cycle++; }
            swaps += cycle - 1;
        }
    }
    free(visited);
    if (swaps % 2 != 0) d = -d;

    matrix_free(L); matrix_free(U); free(piv);
    return d;
}

Matrix* matrix_cholesky(const Matrix *A) {
    if (!A || A->rows != A->cols) {
        fprintf(stderr, "matrix_cholesky: requires square matrix\n");
        return NULL;
    }
    int n = A->rows;
    Matrix *L = matrix_create(n, n);
    if (!L) return NULL;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A->data[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= L->data[i * n + k] * L->data[j * n + k];
            if (i == j) {
                if (sum <= 0.0) {
                    fprintf(stderr, "matrix_cholesky: matrix not positive-definite\n");
                    matrix_free(L);
                    return NULL;
                }
                L->data[i * n + j] = sqrt(sum);
            } else {
                L->data[i * n + j] = sum / L->data[j * n + j];
            }
        }
    }
    return L;
}

Matrix* matrix_cov(const Matrix *X) {
    if (!X) return NULL;
    int n = X->rows;

    /* Centre X: Xc = X - mean(X, axis=0) */
    Vector *mu = matrix_col_mean(X);
    if (!mu) return NULL;
    vector_scale(mu, -1.0);
    Matrix *Xc = matrix_add_row_broadcast(X, mu);
    vector_free(mu);
    if (!Xc) return NULL;

    /* C = Xc^T * Xc  (d x d) */
    Matrix *XcT = matrix_tras(Xc);
    matrix_free(Xc);
    if (!XcT) return NULL;

    mu = matrix_col_mean(X);
    if (!mu) { matrix_free(XcT); return NULL; }
    vector_scale(mu, -1.0);
    Xc = matrix_add_row_broadcast(X, mu);
    vector_free(mu);
    if (!Xc) { matrix_free(XcT); return NULL; }

    Matrix *C = matrix_mul(XcT, Xc);
    matrix_free(XcT);
    matrix_free(Xc);
    if (!C) return NULL;

    double scale = (n > 1) ? 1.0 / (n - 1) : 1.0;
    matrix_scale(C, scale);
    return C;
}

Matrix* matrix_standardise(const Matrix *X) {
    if (!X) return NULL;

    Vector *mu  = matrix_col_mean(X);
    if (!mu) return NULL;

    /* Compute per-column std */
    Vector *sigma = vector_create(X->cols);
    if (!sigma) { vector_free(mu); return NULL; }

    for (int j = 0; j < X->cols; j++) {
        double var = 0.0;
        for (int i = 0; i < X->rows; i++) {
            double d = X->data[i * X->cols + j] - mu->data[j];
            var += d * d;
        }
        sigma->data[j] = (X->rows > 1) ? sqrt(var / (X->rows - 1)) : 1.0;
        if (sigma->data[j] < 1e-12) sigma->data[j] = 1.0; /* avoid / 0 */
    }

    Matrix *out = matrix_create(X->rows, X->cols);
    if (!out) { vector_free(mu); vector_free(sigma); return NULL; }

    for (int i = 0; i < X->rows; i++)
        for (int j = 0; j < X->cols; j++)
            out->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - mu->data[j]) / sigma->data[j];

    vector_free(mu);
    vector_free(sigma);
    return out;
}

Matrix* matrix_minmax_norm(const Matrix *X) {
    if (!X) return NULL;

    Matrix *out = matrix_copy(X);
    if (!out) return NULL;

    for (int j = 0; j < X->cols; j++) {
        double mn = X->data[j], mx = X->data[j];
        for (int i = 1; i < X->rows; i++) {
            double v = X->data[i * X->cols + j];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double range = mx - mn;
        if (range < 1e-12) range = 1.0;
        for (int i = 0; i < X->rows; i++)
            out->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - mn) / range;
    }
    return out;
}

Matrix* matrix_outer(const Vector *u, const Vector *v) {
    if (!u || !v) return NULL;
    Matrix *M = matrix_create(u->size, v->size);
    if (!M) return NULL;
    for (int i = 0; i < u->size; i++)
        for (int j = 0; j < v->size; j++)
            M->data[i * v->size + j] = u->data[i] * v->data[j];
    return M;
}

Matrix* matrix_softmax_rows(const Matrix *m) {
    if (!m) return NULL;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < m->rows; i++) {
        Vector *row = matrix_get_row(m, i);
        if (!row) { matrix_free(out); return NULL; }
        Vector *sm  = vector_softmax(row);
        vector_free(row);
        if (!sm)  { matrix_free(out); return NULL; }
        memcpy(out->data + i * m->cols, sm->data, m->cols * sizeof(double));
        vector_free(sm);
    }
    return out;
}

Matrix* matrix_sigmoid(const Matrix *m) {
    if (!m) return NULL;
    int total = m->rows * m->cols;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < total; i++)
        out->data[i] = 1.0 / (1.0 + exp(-m->data[i]));
    return out;
}

Matrix* matrix_relu(const Matrix *m) {
    if (!m) return NULL;
    int total = m->rows * m->cols;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < total; i++)
        out->data[i] = m->data[i] > 0.0 ? m->data[i] : 0.0;
    return out;
}

Matrix* matrix_tanh_elem(const Matrix *m) {
    if (!m) return NULL;
    int total = m->rows * m->cols;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < total; i++) out->data[i] = tanh(m->data[i]);
    return out;
}

Matrix* matrix_clip(const Matrix *m, double lo, double hi) {
    if (!m) return NULL;
    Matrix *out = matrix_copy(m);
    if (!out) return NULL;
    int total = out->rows * out->cols;
    for (int i = 0; i < total; i++) {
        if (out->data[i] < lo) out->data[i] = lo;
        else if (out->data[i] > hi) out->data[i] = hi;
    }
    return out;
}

Matrix* matrix_log(const Matrix *m) {
    if (!m) return NULL;
    int total = m->rows * m->cols;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < total; i++) out->data[i] = log(m->data[i]);
    return out;
}

Matrix* matrix_exp(const Matrix *m) {
    if (!m) return NULL;
    int total = m->rows * m->cols;
    Matrix *out = matrix_create(m->rows, m->cols);
    if (!out) return NULL;
    for (int i = 0; i < total; i++) out->data[i] = exp(m->data[i]);
    return out;
}

void matrix_print(const Matrix *m) {
    if (!m) { printf("(NULL)\n"); return; }
    for (int i = 0; i < m->rows; i++) {
        printf("[ ");
        for (int j = 0; j < m->cols; j++)
            printf("%10.5f ", m->data[i * m->cols + j]);
        printf("]\n");
    }
}

void matrix_print_named(const char *name, const Matrix *m) {
    printf("%s (%dx%d):\n", name ? name : "M",
           m ? m->rows : -1, m ? m->cols : -1);
    matrix_print(m);
}

void matrix_print_shape(const char *name, const Matrix *m) {
    printf("%s: %dx%d\n",
           name ? name : "M",
           m ? m->rows : -1,
           m ? m->cols : -1);
}