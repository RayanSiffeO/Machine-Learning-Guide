/* ── tree.c ── */
#include "tree.h"
#include "argsort.h"
#include <math.h>
#include <string.h>

static double gini(const int *counts, int n_classes, int total) {
    if (total == 0) return 0.0;
    double g = 1.0;
    for (int c = 0; c < n_classes; c++) {
        double p = (double)counts[c] / total;
        g -= p * p;
    }
    return g;
}

static double entropy(const int *counts, int n_classes, int total) {
    if (total == 0) return 0.0;
    double e = 0.0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] == 0) continue;
        double p = (double)counts[c] / total;
        e -= p * log2(p);
    }
    return e;
}

static double node_impurity(const int *counts, int n_classes,
                             int total, Criterion crit) {
    return (crit == CRITERION_ENTROPY)
        ? entropy(counts, n_classes, total)
        : gini(counts, n_classes, total);
}

/* Varianza para regresión */
static double variance_of(const double *y, const int *idx,
                            int n, double *mean_out) {
    if (n == 0) { if (mean_out) *mean_out = 0.0; return 0.0; }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += y[idx[i]];
    double mean = sum / n;
    if (mean_out) *mean_out = mean;
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = y[idx[i]] - mean;
        var += d * d;
    }
    return var / n;
}

static int node_add(TreeNodes *nodes, int feature, double threshold, double value, int left, int right) {
 
    if (nodes->n_nodes >= nodes->capacity) {
        fprintf(stderr,
            "[tree] FATAL: se alcanzó la capacidad máxima del arena "
            "(%d nodos). Aumenta max_depth o la capacidad inicial.\n",
            nodes->capacity);
        abort();
    }
    int id = nodes->n_nodes++;
    nodes->feature[id]   = feature;
    nodes->threshold[id] = threshold;
    nodes->value[id]     = value;
    nodes->left[id]      = left;
    nodes->right[id]     = right;
    return id;
}


typedef struct {
    int    feature;
    double threshold;
    double gain;  
} BestSplit;

static BestSplit find_best_split(
        const Matrix *X, const double *y,
        const int *sample_idx, int n,
        int *feature_subset, int m,
        int n_classes, Criterion criterion,
        int min_samples_leaf)
{
    BestSplit best = { -1, 0.0, -1.0 };

    /* Counts del nodo padre */
    int *parent_counts = ML_NEW_ARRAY(int, n_classes);
    for (int i = 0; i < n; i++)
        parent_counts[(int)round(y[sample_idx[i]])]++;
    double parent_imp = node_impurity(parent_counts, n_classes, n, criterion);
    ml_free(parent_counts);

    /* Arrays de trabajo reutilizables */
    int    *sort_idx    = ML_NEW_ARRAY(int, n);
    double *feat_vals   = ML_NEW_ARRAY(double, n);
    int    *left_counts = ML_NEW_ARRAY(int, n_classes);
    int    *right_counts= ML_NEW_ARRAY(int, n_classes);

    for (int fi = 0; fi < m; fi++) {
        int f = feature_subset[fi];

        for (int i = 0; i < n; i++) {
            feat_vals[i] = X->data[sample_idx[i] * X->cols + f];
            sort_idx[i]  = i;
        }

        argsort_double(feat_vals, n, sort_idx);

        memset(left_counts,  0, n_classes * sizeof(int));
        memset(right_counts, 0, n_classes * sizeof(int));
        for (int i = 0; i < n; i++)
            right_counts[(int)round(y[sample_idx[sort_idx[i]]])]++;

        for (int i = 0; i < n - 1; i++) {
            int cls = (int)round(y[sample_idx[sort_idx[i]]]);
            left_counts[cls]++;
            right_counts[cls]--;

            int n_left  = i + 1;
            int n_right = n - n_left;

            if (n_left < min_samples_leaf || n_right < min_samples_leaf)
                continue;

            /* Saltar si el umbral es el mismo que el siguiente */
            if (feat_vals[sort_idx[i]] == feat_vals[sort_idx[i+1]])
                continue;

            double imp_left  = node_impurity(left_counts,  n_classes, n_left,  criterion);
            double imp_right = node_impurity(right_counts, n_classes, n_right, criterion);

            double gain = parent_imp
                        - ((double)n_left  / n) * imp_left
                        - ((double)n_right / n) * imp_right;

            if (gain > best.gain) {
                best.gain      = gain;
                best.feature   = f;

                best.threshold = (feat_vals[sort_idx[i]] +
                                  feat_vals[sort_idx[i+1]]) * 0.5;
            }
        }
    }

    ml_free(sort_idx);
    ml_free(feat_vals);
    ml_free(left_counts);
    ml_free(right_counts);

    return best;
}

static int build_node(
        DecisionTree *t,
        const Matrix *X, const double *y,
        int *sample_idx, int n,
        int depth, MT19937 *rng)
{
    int n_classes = t->n_classes;


    double leaf_value;
    if (n_classes > 0) {
        /* Clasificación: clase con más muestras */
        int *counts = ML_NEW_ARRAY(int, n_classes);
        for (int i = 0; i < n; i++)
            counts[(int)round(y[sample_idx[i]])]++;
        int best = 0;
        for (int c = 1; c < n_classes; c++)
            if (counts[c] > counts[best]) best = c;
        leaf_value = (double)best;
        ml_free(counts);
    } else {
        /* Regresión: media */
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += y[sample_idx[i]];
        leaf_value = sum / n;
    }

    /* ── Condiciones de parada → hoja ── */
    int make_leaf = (n < t->min_samples_split)
                 || (t->max_depth >= 0 && depth >= t->max_depth);

    /* También es hoja si todas las muestras son de la misma clase */
    if (!make_leaf && n_classes > 0) {
        int first_cls = (int)round(y[sample_idx[0]]);
        int pure = 1;
        for (int i = 1; i < n; i++)
            if ((int)round(y[sample_idx[i]]) != first_cls) { pure = 0; break; }
        if (pure) make_leaf = 1;
    }

    if (make_leaf)
        return node_add(t->nodes, -1, 0.0, leaf_value, -1, -1);

    int p = X->cols;
    int m = t->max_features > 0 ? t->max_features : (int)sqrt(p);
    if (m > p) m = p;

    int *all_features = ML_NEW_ARRAY(int, p);
    for (int i = 0; i < p; i++) all_features[i] = i;
    sample_without_replacement(rng, p, m, all_features);

    BestSplit best = find_best_split(
        X, y, sample_idx, n,
        all_features, m,
        n_classes, t->criterion,
        t->min_samples_leaf > 0 ? t->min_samples_leaf : 1
    );
    ml_free(all_features);

    if (best.feature < 0 || best.gain <= 0.0)
        return node_add(t->nodes, -1, 0.0, leaf_value, -1, -1);

    if (t->feature_importance)
        t->feature_importance[best.feature] += best.gain * n;

    int left_n  = 0;
    int right_n = 0;
    int *left_idx  = ML_NEW_ARRAY(int, n);
    int *right_idx = ML_NEW_ARRAY(int, n);

    for (int i = 0; i < n; i++) {
        double val = X->data[sample_idx[i] * X->cols + best.feature];
        if (val <= best.threshold)
            left_idx[left_n++]   = sample_idx[i];
        else
            right_idx[right_n++] = sample_idx[i];
    }
    int node_id = node_add(t->nodes, best.feature,
                            best.threshold, leaf_value, -1, -1);

    int left_id  = build_node(t, X, y, left_idx,  left_n,  depth+1, rng);
    int right_id = build_node(t, X, y, right_idx, right_n, depth+1, rng);

    t->nodes->left[node_id]  = left_id;
    t->nodes->right[node_id] = right_id;

    ml_free(left_idx);
    ml_free(right_idx);

    return node_id;
}

DecisionTree* tree_create(int max_depth, int min_samples_split,
                           int min_samples_leaf, int max_features,
                           Criterion criterion, int n_classes) {
    DecisionTree *t = ML_NEW(DecisionTree);
    if (!t) return NULL;

    t->max_depth         = max_depth;
    t->min_samples_split = min_samples_split;
    t->min_samples_leaf  = min_samples_leaf > 0 ? min_samples_leaf : 1;
    t->max_features      = max_features;
    t->criterion         = criterion;
    t->n_classes         = n_classes;
    t->feature_importance = NULL;
    t->n_features        = 0;

    t->arena = arena_create(ML_ARENA_DEFAULT_CAPACITY);
    if (!t->arena) { ml_free(t); return NULL; }

    t->nodes = arena_alloc(t->arena, sizeof(TreeNodes));
    if (!t->nodes) { arena_free(t->arena); ml_free(t); return NULL; }

    int cap;
    if (max_depth >= 0 && max_depth < 20)
        cap = (1 << (max_depth + 1)) * 2; 
    else
        cap = 65536;                

    t->nodes->capacity  = cap;
    t->nodes->n_nodes   = 0;
    t->nodes->feature   = arena_alloc(t->arena, sizeof(int)    * cap);
    t->nodes->threshold = arena_alloc(t->arena, sizeof(double) * cap);
    t->nodes->left      = arena_alloc(t->arena, sizeof(int)    * cap);
    t->nodes->right     = arena_alloc(t->arena, sizeof(int)    * cap);
    t->nodes->value     = arena_alloc(t->arena, sizeof(double) * cap);

    return t;
}

void tree_free(DecisionTree *t) {
    if (!t) return;
    if (t->feature_importance) ml_free(t->feature_importance);
    if (t->arena) arena_free(t->arena);
    ml_free(t);
}

void tree_fit(DecisionTree *t, const Matrix *X, const Vector *y,
              MT19937 *rng, int *sample_indices, int n_samples) {
    if (!t || !X || !y || !rng) return;

    t->n_features = X->cols;
    t->feature_importance = ML_NEW_ARRAY(double, t->n_features);

    t->root = build_node(t, X, y->data, sample_indices, n_samples, 0, rng);
}

double tree_predict_one(const DecisionTree *t, const double *x) {
    if (!t || !x) return NAN;
    int node = t->root;
    while (t->nodes->left[node] != -1) {
        int f = t->nodes->feature[node];
        node = (x[f] <= t->nodes->threshold[node])
             ? t->nodes->left[node]
             : t->nodes->right[node];
    }
    return t->nodes->value[node];
}

Vector* tree_predict(const DecisionTree *t, const Matrix *X) {
    if (!t || !X) return NULL;
    Vector *out = vector_create(X->rows);
    if (!out) return NULL;
    for (int i = 0; i < X->rows; i++)
        out->data[i] = tree_predict_one(t, X->data + i * X->cols);
    return out;
}