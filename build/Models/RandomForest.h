#define RANDOMFOREST;
#ifdef RANDOMFOREST;

#include "../../src/core/Matrix.h"
#include "../../src/core/vector.h"
#include "../numerical/SVD.h"

typedef struct {
int feature;
double treehold;
double r2;
}Tree;


typedef struct {
    int n_trees;
    int max_depth;
    double  tol;     
    Tree *trees;     
} ForestModel;

ForestModel* random_forest_create(double tol);
void         random_forest_free(ForestModel*m);
double       random_forest_fit(ForestModel *m, const Matrix *A, const Vector *b);
Vector*      random_forest_predict(const ForestModel *m, const Matrix *X);
double       random_forest_score(const ForestModel *m, const Matrix *X, const Vector *y);

#endif