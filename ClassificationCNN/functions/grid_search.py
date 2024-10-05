from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search(X, y, model):
    param_grid = {
        'num_epochs': [500, 700, 1100],
        'patience': [55, 75, 100],
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0.0),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0.0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0.0),
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit=False, verbose=3)
    grid_search.fit(X, y)
    return grid_search

