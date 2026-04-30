#include "RandomForest.h"
#include <math.h>

static int compute_n_classes(const Vector *y) {
    int max_cls = 0;
    for (int i = 0; i < y->size; i++) {
        int cls = (int)round(y->data[i]);
        if (cls < 0) {
            fprintf(stderr,
                "[rf] WARN: etiqueta negativa %d en índice %d — "
                "se ignorará en el conteo de clases.\n", cls, i);
            continue;
        }
        if (cls > max_cls) max_cls = cls;
    }
    return max_cls + 1;
}

RandomForest* rf_create(int n_estimators, int max_depth,
                         int min_samples_split, int min_samples_leaf,
                         int max_features,
                         Criterion criterion, uint32_t random_state) {
    RandomForest *rf = ML_NEW(RandomForest);
    if (!rf) return NULL;

    rf->n_estimators      = n_estimators;
    rf->max_depth         = max_depth;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf  = min_samples_leaf > 0 ? min_samples_leaf : 1;
    rf->max_features      = max_features;
    rf->criterion         = criterion;
    rf->random_state      = random_state;
    rf->bootstrap         = 1;
    rf->n_trees_fitted    = 0;
    rf->n_features        = 0;
    rf->n_classes         = 0;
    rf->feature_importances = NULL;
    rf->oob_enabled       = 0;
    rf->oob_score         = 0.0;

    rf->trees = ML_NEW_ARRAY(DecisionTree*, n_estimators);
    rf->rngs  = ML_NEW_ARRAY(MT19937, n_estimators);

    if (!rf->trees || !rf->rngs) { rf_free(rf); return NULL; }

    MT19937 master;
    mt_init(&master, random_state);
    for (int i = 0; i < n_estimators; i++)
        mt_init(&rf->rngs[i], mt_next(&master));

    return rf;
}

void rf_free(RandomForest *rf) {
    if (!rf) return;
    if (rf->trees) {
        for (int i = 0; i < rf->n_trees_fitted; i++)
            tree_free(rf->trees[i]);
        ml_free(rf->trees);
    }
    ml_free(rf->rngs);
    ml_free(rf->feature_importances);
    ml_free(rf);
}

void rf_fit(RandomForest *rf, const Matrix *X, const Vector *y) {
    if (!rf || !X || !y) return;

    rf->n_features = X->cols;
    /* FIX (n_classes frágil): usar helper robusto en lugar de vector_max */
    rf->n_classes  = compute_n_classes(y);

    if (rf->max_features <= 0)
        rf->max_features = (int)sqrt((double)rf->n_features);
    rf->feature_importances = ML_NEW_ARRAY(double, rf->n_features);
    if (!rf->feature_importances) return;

    int n = X->rows;
    int *indices = ML_NEW_ARRAY(int, n);
    if (!indices) return;

    for (int i = 0; i < rf->n_estimators; i++) {

        rf->trees[i] = tree_create(
            rf->max_depth,
            rf->min_samples_split,
            rf->min_samples_leaf,
            rf->max_features,
            rf->criterion,
            rf->n_classes
        );
        if (!rf->trees[i]) continue;

        if (rf->bootstrap)
            bootstrap_sample(&rf->rngs[i], n, indices);
        else
            for (int j = 0; j < n; j++) indices[j] = j;

        tree_fit(rf->trees[i], X, y, &rf->rngs[i], indices, n);

        if (rf->trees[i]->feature_importance) {
            for (int f = 0; f < rf->n_features; f++)
                rf->feature_importances[f] +=
                    rf->trees[i]->feature_importance[f];
        }

        rf->n_trees_fitted++;
    }

    ml_free(indices);

    if (rf->n_trees_fitted > 0) {
        double total = 0.0;
        for (int f = 0; f < rf->n_features; f++)
            total += rf->feature_importances[f];
        if (total > 0.0)
            for (int f = 0; f < rf->n_features; f++)
                rf->feature_importances[f] /= total;
    }
}

Vector* rf_predict(const RandomForest *rf, const Matrix *X) {
    if (!rf || !X || rf->n_trees_fitted == 0) return NULL;

    int n = X->rows;
    Vector *out = vector_create(n);
    if (!out) return NULL;

    int *votes = ML_NEW_ARRAY(int, n * rf->n_classes);
    if (!votes) { vector_free(out); return NULL; }

    for (int t = 0; t < rf->n_trees_fitted; t++) {
        for (int i = 0; i < n; i++) {
            const double *row = X->data + i * X->cols;
            int pred = (int)round(tree_predict_one(rf->trees[t], row));
            if (pred >= 0 && pred < rf->n_classes)
                votes[i * rf->n_classes + pred]++;
        }
    }

    for (int i = 0; i < n; i++) {
        int best_class = 0, best_votes = 0;
        for (int c = 0; c < rf->n_classes; c++) {
            if (votes[i * rf->n_classes + c] > best_votes) {
                best_votes = votes[i * rf->n_classes + c];
                best_class = c;
            }
        }
        out->data[i] = (double)best_class;
    }

    ml_free(votes);
    return out;
}

Matrix* rf_predict_proba(const RandomForest *rf, const Matrix *X) {
    if (!rf || !X || rf->n_trees_fitted == 0) return NULL;

    int n = X->rows;
    Matrix *proba = matrix_create(n, rf->n_classes);
    if (!proba) return NULL;

    for (int t = 0; t < rf->n_trees_fitted; t++) {
        for (int i = 0; i < n; i++) {
            const double *row = X->data + i * X->cols;
            int pred = (int)round(tree_predict_one(rf->trees[t], row));
            if (pred >= 0 && pred < rf->n_classes)
                proba->data[i * rf->n_classes + pred] += 1.0;
        }
    }

    matrix_scale(proba, 1.0 / rf->n_trees_fitted);
    return proba;
}

double rf_score(const RandomForest *rf, const Matrix *X, const Vector *y) {
    Vector *pred = rf_predict(rf, X);
    if (!pred) return NAN;
    double score = accuracy_metric(y, pred);
    vector_free(pred);
    return score;
}

Vector* rf_feature_importances(const RandomForest *rf) {
    if (!rf || !rf->feature_importances) return NULL;
    return vector_from_array(rf->feature_importances, rf->n_features);
}

double rf_oob_score(const RandomForest *rf) {
    if (!rf || rf->n_trees_fitted == 0 || !rf->bootstrap) return NAN;
    fprintf(stderr,
        "[rf] WARN: rf_oob_score requiere guardar índices OOB durante fit. "
        "Activa rf->oob_enabled=1 antes de llamar a rf_fit.\n");
    return rf->oob_score;  
}