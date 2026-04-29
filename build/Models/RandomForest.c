#include "RandomForest.h"
#include <stdio.h>
#include <stdlib.h>

ForestModel* random_forest_create(double tol) {
    ForestModel *m = (ForestModel *)calloc(1, sizeof(ForestModel));
    if (!m) return NULL;
    m->r2  = 0.0;
    m -> tol = tol;
    return m;
}

void random_forest_free(ForestModel*m) {
    if (!m) return;
    vector_free(m -> weights);  
    free(m);
}

