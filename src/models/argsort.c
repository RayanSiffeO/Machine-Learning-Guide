#define _GNU_SOURCE  
#include <stdlib.h>
#include "argsort.h"

typedef struct {
    const double *arr;
} ArgsortCtx;

static int cmp_idx(const void *a, const void *b, void *ctx) {
    const ArgsortCtx *c = (const ArgsortCtx *)ctx;
    double da = c->arr[*(const int *)a];
    double db = c->arr[*(const int *)b];
    return (da > db) - (da < db);
}

void argsort_double(const double *arr, int n, int *indices) {
    for (int i = 0; i < n; i++) indices[i] = i;
    ArgsortCtx ctx = { arr };
    qsort_r(indices, (size_t)n, sizeof(int), cmp_idx, &ctx);
}