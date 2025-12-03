#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score

# Load datasets
data = pd.read_csv("train_FuelConsumption.csv")
test_data = pd.read_csv("test_simul_FuelConsumption.csv")

# Features and target
num_cols = data.select_dtypes(include='number').columns
features = [col for col in num_cols if col != 'CO2EMISSIONS']
X = data[features]
y = data['CO2EMISSIONS']
X_test = test_data[features]
y_test = test_data['CO2EMISSIONS']

# GridSearchCV para otimizar RandomForest
print("=== Otimizando RandomForest com GridSearchCV ===")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

# Resultados do GridSearchCV
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R2 score: {grid_search.best_score_:.4f}")

# Cross-validation detalhada do melhor modelo
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"\nCross-validation scores (5 folds): {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Std CV score: {cv_scores.std():.4f}")

# Avaliar no test_simul
y_pred = grid_search.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score on Test_Simul: {r2:.4f}")

# %%