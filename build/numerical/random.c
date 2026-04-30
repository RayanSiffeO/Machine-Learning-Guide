#include "random.h"
#include "../core/memory.h"
#include <string.h>

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER    0x80000000UL
#define MT_LOWER    0x7fffffffUL

void mt_init(MT19937 *rng, uint32_t seed) {
    rng->mt[0] = seed;
    for (int i = 1; i < MT_N; i++)
        rng->mt[i] = 1812433253UL *
                     (rng->mt[i-1] ^ (rng->mt[i-1] >> 30)) + i;
    rng->index = MT_N;
}

uint32_t mt_next(MT19937 *rng) {
    static const uint32_t mag[2] = {0, MT_MATRIX_A};

    if (rng->index >= MT_N) {
        int i;
        uint32_t y;
        for (i = 0; i < MT_N - MT_M; i++) {
            y = (rng->mt[i] & MT_UPPER) | (rng->mt[i+1] & MT_LOWER);
            rng->mt[i] = rng->mt[i + MT_M] ^ (y >> 1) ^ mag[y & 1];
        }
        for (; i < MT_N - 1; i++) {
            y = (rng->mt[i] & MT_UPPER) | (rng->mt[i+1] & MT_LOWER);
            rng->mt[i] = rng->mt[i + (MT_M - MT_N)] ^ (y >> 1) ^ mag[y & 1];
        }
        y = (rng->mt[MT_N-1] & MT_UPPER) | (rng->mt[0] & MT_LOWER);
        rng->mt[MT_N-1] = rng->mt[MT_M-1] ^ (y >> 1) ^ mag[y & 1];
        rng->index = 0;
    }

    uint32_t y = rng->mt[rng->index++];
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

double mt_uniform(MT19937 *rng) {
    return (mt_next(rng) >> 1) * (1.0 / 2147483648.0);
}

int mt_randint(MT19937 *rng, int n) {
    return (int)(mt_uniform(rng) * n);
}

void bootstrap_sample(MT19937 *rng, int n, int *out) {
    for (int i = 0; i < n; i++)
        out[i] = mt_randint(rng, n);
}

void shuffle_int(MT19937 *rng, int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = mt_randint(rng, i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/*
 * FIX: La versión original usaba malloc/free directos, escapando al sistema
 * ml_malloc/ml_free y rompiendo el tracking en modo ML_DEBUG_MEMORY.
 * Ahora usa ml_malloc/ml_free como el resto del proyecto.
 */
void sample_without_replacement(MT19937 *rng, int n, int k, int *out) {
    int *pool = (int *)ml_malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) pool[i] = i;
    for (int i = 0; i < k; i++) {
        int j = i + mt_randint(rng, n - i);
        int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        out[i] = pool[i];
    }
    ml_free(pool);
}