#ifndef TREE_H
#define TREE_H

#include "../core/Matrix.h"
#include "../core/vector.h"
#include "random.h"
#include "../core/memory.h"

typedef enum { CRITERION_GINI, CRITERION_ENTROPY, CRITERION_MSE } Criterion;
typedef struct {
    int    *feature;     
    double *threshold;    
    double *impurity;   
    int    *n_samples; 
    int    *left;        
    int    *right;      
    double *value;        
    int     n_nodes;
    int     capacity;
} TreeNodes;

typedef struct {
    TreeNodes  *nodes;
    int         root;
    int         max_depth;
    int         min_samples_split;
    int         min_samples_leaf;
    int         max_features;   
    Criterion   criterion;
    int         n_features;
    int         n_classes;  
    double     *feature_importance; 
    MlArena    *arena;
} DecisionTree;

DecisionTree* tree_create(int max_depth, int min_samples_split, int min_samples_leaf, int max_features, Criterion criterion, int n_classes);
void tree_free(DecisionTree *t);
void tree_fit(DecisionTree *t, const Matrix *X, const Vector *y, MT19937 *rng, int *sample_indices, int n_samples);
double tree_predict_one(const DecisionTree *t, const double *x);
Vector* tree_predict(const DecisionTree *t, const Matrix *X);

#endif