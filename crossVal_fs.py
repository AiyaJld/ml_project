from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import numpy as np
import pandas as pd

def forward_selection(X, y, model, max_features=15):
    selected_features = []
    remaining_features = list(X.columns)
    best_score = np.inf
    while len(selected_features) < max_features and remaining_features:
        scores = []
        for feature in remaining_features:
            try:
                model_clone = clone(model)       #cloned clear model not to overtrain 
                model_clone.fit(X[selected_features + [feature]], y)
                preds = model_clone.predict(X[selected_features + [feature]])
                rmse = mean_squared_error(y, preds) ** 0.5
                scores.append((rmse, feature))
            except Exception as e:
                continue  

        if not scores:
            break
        scores.sort()
        #check if the score is better then prev best score
        if scores[0][0] < best_score:
            best_score, best_feature = scores[0]
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return selected_features

def evaluate_model_with_cv(X, y, model, n_splits=5, max_features=15, min_votes=4):
    MODELS_WITHOUT_FS = (RandomForestRegressor, DecisionTreeRegressor)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    all_selected = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)

        # Feature selection — that is not RandomForest или DecisionTree
        if not isinstance(model, MODELS_WITHOUT_FS):
            selected_features = forward_selection(X_train_scaled, y_train, model, max_features=max_features)
        else:
            selected_features = list(X.columns)

        all_selected.append(selected_features)

        model_clone = clone(model)
        model_clone.fit(X_train_scaled[selected_features], y_train)
        preds = model_clone.predict(X_val_scaled[selected_features])
        rmse = mean_squared_error(y_val, preds) ** 0.5
        rmse_scores.append(rmse)

        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
        flat_list = [f for fold_feats in all_selected for f in fold_feats]

    feature_counts = Counter(flat_list)
    final_features = [feature for feature, count in feature_counts.items() if count >= min_votes]

    avg_rmse = np.mean(rmse_scores)

    return {
        "avg_rmse": avg_rmse,
        "rmse_per_fold": rmse_scores,
        "final_features": final_features
    }