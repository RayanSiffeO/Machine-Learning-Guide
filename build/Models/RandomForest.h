#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "tree.h"
#include "../core/Matrix.h"
#include "../core/vector.h"
#include "../src/core/memory.h"

typedef struct {
    int       n_estimators;
    int       max_depth;
    int       min_samples_split;
    int       min_samples_leaf;
    int       max_features;    
    Criterion criterion;
    int       bootstrap;          
    uint32_t  random_state;
    DecisionTree **trees;
    int            n_trees_fitted;

    int     n_features;
    int     n_classes;  
    double *feature_importances;
    int     oob_enabled;
    double  oob_score;
    MT19937 *rngs;
} RandomForest;
RandomForest* rf_create(int n_estimators, int max_depth, int min_samples_split, int min_samples_leaf, int max_features, Criterion criterion, uint32_t random_state);
void          rf_free(RandomForest *rf);
void          rf_fit(RandomForest *rf, const Matrix *X, const Vector *y);
Vector*       rf_predict(const RandomForest *rf, const Matrix *X);
Matrix*       rf_predict_proba(const RandomForest *rf, const Matrix *X);
Vector*       rf_feature_importances(const RandomForest *rf);
double        rf_oob_score(const RandomForest *rf);
double        rf_score(const RandomForest *rf, const Matrix *X, const Vector *y); 

#endif