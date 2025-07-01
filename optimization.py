from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def opt_tree(x_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
    }
    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=5,
        n_jobs=-1,           
        verbose=2
    )
    grid_search.fit(x_train, y_train)
    print("Best:", grid_search.best_params_)
    print("Best R² score:", grid_search.best_score_)
    return grid_search.best_estimator_, grid_search.best_params_

def opt_rf(x_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_distributions = {
        'n_estimators': np.arange(100, 1001, 100),            
        'max_depth': [None] + list(np.arange(5, 30, 5)),      
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],   
        'bootstrap': [True, False],
    }
    # 3. RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=30,                  
        cv=5,                     
        verbose=2,
        n_jobs=-1,                 
        random_state=42,
        scoring='r2'              
    )

    random_search.fit(x_train, y_train)


    print("Best:", random_search.best_params_)
    return random_search.best_params_