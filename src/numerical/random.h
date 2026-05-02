#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#include "../core/vector.h"
#include <stdlib.h>

#define MT_N 624

typedef struct {
    uint32_t mt[624];
    int      index;
} MT19937;

void     mt_init(MT19937 *rng, uint32_t seed);
uint32_t mt_next(MT19937 *rng);
double   mt_uniform(MT19937 *rng);      
int      mt_randint(MT19937 *rng, int n);  


void bootstrap_sample(MT19937 *rng, int n, int *out);
void sample_without_replacement(MT19937 *rng, int n, int k, int *out);
void shuffle_int(MT19937 *rng, int *arr, int n);

#endif