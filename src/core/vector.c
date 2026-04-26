#include "vector.h"

int vector_check_dims(const Vector *a, const Vector *b, const char *fn) {
    if (!a || !b || a->size != b->size) {
        fprintf(stderr, "%s: dimension mismatch (%d vs %d)\n",
                fn,
                a ? a->size : -1,
                b ? b->size : -1);
        return 0;
    }
    return 1;
}

Vector* vector_create(int size) {
    if (size <= 0) return NULL;

    Vector *v = (Vector*)malloc(sizeof(Vector));
    if (!v) return NULL;

    v->size = size;
    v->data = (double*)calloc(size, sizeof(double));
    if (!v->data) { free(v); return NULL; }

    return v;
}

void vector_free(Vector *v) {
    if (v) {
        free(v->data);
        free(v);
    }
}

Vector* vector_copy(const Vector *src) {
    if (!src) return NULL;

    Vector *dst = vector_create(src->size);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->size * sizeof(double));
    return dst;
}

Vector* vector_from_array(const double *arr, int size) {
    if (!arr || size <= 0) return NULL;

    Vector *v = vector_create(size);
    if (!v) return NULL;

    memcpy(v->data, arr, size * sizeof(double));
    return v;
}

void vector_fill(Vector *v, double value) {
    if (!v) return;
    for (int i = 0; i < v->size; i++) v->data[i] = value;
}

void vector_zero(Vector *v) {
    if (v) memset(v->data, 0, v->size * sizeof(double));
}


double vector_get(const Vector *v, int i) {
    if (!v || i < 0 || i >= v->size) {
        fprintf(stderr, "vector_get: index %d out of range (size %d)\n",
                i, v ? v->size : -1);
        return NAN;
    }
    return v->data[i];
}

void vector_set(Vector *v, int i, double value) {
    if (!v || i < 0 || i >= v->size) {
        fprintf(stderr, "vector_set: index %d out of range (size %d)\n",
                i, v ? v->size : -1);
        return;
    }
    v->data[i] = value;
}

Vector* vector_add(const Vector *a, const Vector *b) {
    if (!vector_check_dims(a, b, "vector_add")) return NULL;

    Vector *c = vector_create(a->size);
    if (!c) return NULL;

    for (int i = 0; i < a->size; i++)
        c->data[i] = a->data[i] + b->data[i];
    return c;
}

Vector* vector_sub(const Vector *a, const Vector *b) {
    if (!vector_check_dims(a, b, "vector_sub")) return NULL;

    Vector *c = vector_create(a->size);
    if (!c) return NULL;

    for (int i = 0; i < a->size; i++)
        c->data[i] = a->data[i] - b->data[i];
    return c;
}

Vector* vector_mul_elem(const Vector *a, const Vector *b) {
    if (!vector_check_dims(a, b, "vector_mul_elem")) return NULL;

    Vector *c = vector_create(a->size);
    if (!c) return NULL;

    for (int i = 0; i < a->size; i++)
        c->data[i] = a->data[i] * b->data[i];
    return c;
}

Vector* vector_div_elem(const Vector *a, const Vector *b) {
    if (!vector_check_dims(a, b, "vector_div_elem")) return NULL;

    Vector *c = vector_create(a->size);
    if (!c) return NULL;

    for (int i = 0; i < a->size; i++) {
        if (b->data[i] == 0.0) {
            fprintf(stderr, "vector_div_elem: division by zero at index %d\n", i);
            vector_free(c);
            return NULL;
        }
        c->data[i] = a->data[i] / b->data[i];
    }
    return c;
}

void vector_scale(Vector *v, double scalar) {
    if (!v) return;
    for (int i = 0; i < v->size; i++) v->data[i] *= scalar;
}

void vector_shift(Vector *v, double scalar) {
    if (!v) return;
    for (int i = 0; i < v->size; i++) v->data[i] += scalar;
}

void vector_axpy(Vector *v, double alpha, const Vector *u) {
    if (!vector_check_dims(v, u, "vector_axpy")) return;
    for (int i = 0; i < v->size; i++)
        v->data[i] += alpha * u->data[i];
}

double vector_dot(const Vector *a, const Vector *b) {
    if (!vector_check_dims(a, b, "vector_dot")) return NAN;

    double acc = 0.0;
    for (int i = 0; i < a->size; i++) acc += a->data[i] * b->data[i];
    return acc;
}

double vector_norm(const Vector *v) {
    if (!v) return NAN;
    return sqrt(vector_dot(v, v));
}

double vector_norm1(const Vector *v) {
    if (!v) return NAN;
    double acc = 0.0;
    for (int i = 0; i < v->size; i++) acc += fabs(v->data[i]);
    return acc;
}

double vector_norm_inf(const Vector *v) {
    if (!v) return NAN;
    double mx = 0.0;
    for (int i = 0; i < v->size; i++) {
        double a = fabs(v->data[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

double vector_sum(const Vector *v) {
    if (!v) return NAN;
    double acc = 0.0;
    for (int i = 0; i < v->size; i++) acc += v->data[i];
    return acc;
}

double vector_mean(const Vector *v) {
    if (!v || v->size == 0) return NAN;
    return vector_sum(v) / v->size;
}

double vector_variance(const Vector *v) {
    if (!v || v->size == 0) return NAN;

    double mu = vector_mean(v);
    double acc = 0.0;
    for (int i = 0; i < v->size; i++) {
        double d = v->data[i] - mu;
        acc += d * d;
    }
    return acc / v->size;
}

double vector_std(const Vector *v) {
    return sqrt(vector_variance(v));
}

double vector_min(const Vector *v) {
    if (!v || v->size == 0) return NAN;
    double m = v->data[0];
    for (int i = 1; i < v->size; i++)
        if (v->data[i] < m) m = v->data[i];
    return m;
}

double vector_max(const Vector *v) {
    if (!v || v->size == 0) return NAN;
    double m = v->data[0];
    for (int i = 1; i < v->size; i++)
        if (v->data[i] > m) m = v->data[i];
    return m;
}

int vector_argmin(const Vector *v) {
    if (!v || v->size == 0) return -1;
    int idx = 0;
    for (int i = 1; i < v->size; i++)
        if (v->data[i] < v->data[idx]) idx = i;
    return idx;
}

int vector_argmax(const Vector *v) {
    if (!v || v->size == 0) return -1;
    int idx = 0;
    for (int i = 1; i < v->size; i++)
        if (v->data[i] > v->data[idx]) idx = i;
    return idx;
}

void vector_normalize(Vector *v) {
    if (!v) return;
    double n = vector_norm(v);
    if (n > 0.0) vector_scale(v, 1.0 / n);
}

Vector* vector_clip(const Vector *v, double lo, double hi) {
    if (!v) return NULL;
    Vector *c = vector_copy(v);
    if (!c) return NULL;
    for (int i = 0; i < c->size; i++) {
        if (c->data[i] < lo) c->data[i] = lo;
        else if (c->data[i] > hi) c->data[i] = hi;
    }
    return c;
}

Vector* vector_log(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = log(v->data[i]);
    return c;
}

Vector* vector_exp(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = exp(v->data[i]);
    return c;
}

Vector* vector_sqrt(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = sqrt(v->data[i]);
    return c;
}

Vector* vector_pow(const Vector *v, double p) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = pow(v->data[i], p);
    return c;
}

Vector* vector_abs(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = fabs(v->data[i]);
    return c;
}

Vector* vector_softmax(const Vector *v) {
    if (!v) return NULL;

    double mx = vector_max(v);          /* numerical stability */
    Vector *c = vector_create(v->size);
    if (!c) return NULL;

    double sum = 0.0;
    for (int i = 0; i < v->size; i++) {
        c->data[i] = exp(v->data[i] - mx);
        sum += c->data[i];
    }
    for (int i = 0; i < v->size; i++) c->data[i] /= sum;
    return c;
}

Vector* vector_sigmoid(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++)
        c->data[i] = 1.0 / (1.0 + exp(-v->data[i]));
    return c;
}

Vector* vector_relu(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++)
        c->data[i] = v->data[i] > 0.0 ? v->data[i] : 0.0;
    return c;
}

Vector* vector_leaky_relu(const Vector *v, double alpha) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++)
        c->data[i] = v->data[i] > 0.0 ? v->data[i] : alpha * v->data[i];
    return c;
}

Vector* vector_tanh(const Vector *v) {
    if (!v) return NULL;
    Vector *c = vector_create(v->size);
    if (!c) return NULL;
    for (int i = 0; i < v->size; i++) c->data[i] = tanh(v->data[i]);
    return c;
}

Vector* vector_one_hot(int class_idx, int n_classes) {
    if (class_idx < 0 || class_idx >= n_classes || n_classes <= 0) return NULL;
    Vector *v = vector_create(n_classes);
    if (!v) return NULL;
    v->data[class_idx] = 1.0;
    return v;
}

Vector* vector_slice(const Vector *v, int start, int end) {
    if (!v || start < 0 || end > v->size || start >= end) {
        fprintf(stderr, "vector_slice: invalid range [%d,%d) for size %d\n",
                start, end, v ? v->size : -1);
        return NULL;
    }
    int len = end - start;
    Vector *s = vector_create(len);
    if (!s) return NULL;
    memcpy(s->data, v->data + start, len * sizeof(double));
    return s;
}

Vector* vector_concat(const Vector *a, const Vector *b) {
    if (!a || !b) return NULL;
    int total = a->size + b->size;
    Vector *c = vector_create(total);
    if (!c) return NULL;
    memcpy(c->data,          a->data, a->size * sizeof(double));
    memcpy(c->data + a->size, b->data, b->size * sizeof(double));
    return c;
}

void vector_print(const Vector *v) {
    if (!v) { printf("[NULL]\n"); return; }
    printf("[");
    for (int i = 0; i < v->size; i++) {
        printf("%.6g", v->data[i]);
        if (i < v->size - 1) printf(", ");
    }
    printf("]\n");
}

void vector_print_named(const char *name, const Vector *v) {
    printf("%s = ", name ? name : "v");
    vector_print(v);
}