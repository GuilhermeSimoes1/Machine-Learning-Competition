#%%
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from joblib import dump

# Dar load nos datasets
data = pd.read_csv("train_FuelConsumption.csv")
test_data = pd.read_csv("test_simul_FuelConsumption.csv")

features = [
    'ENGINESIZE',
    'CYLINDERS',
    'FUELCONSUMPTION_CITY',
    'FUELCONSUMPTION_HWY',
    'FUELCONSUMPTION_COMB',
    'FUELCONSUMPTION_COMB_MPG'
]
X = data[features]
y = data['CO2EMISSIONS']
X_test = test_data[features]
y_test = test_data['CO2EMISSIONS']

# Definir modelos e parâmetros
models = [
    {
        'name': 'Ridge',
        'pipeline': Pipeline([
            ('poly', PolynomialFeatures()),
            ('regressor', Ridge())
        ]),
        'params': {
            'poly__degree': [2, 3, 4],
            'regressor__alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
        }
    },
    {
        'name': 'RandomForest',
        'pipeline': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    }
]

# Treinar, comparar e guardar modelos (o grid já faz cross-validation por si só)
results = []

for model_config in models:
    
    grid_search = GridSearchCV(
        model_config['pipeline'],
        model_config['params'],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Avaliar no test_simul
    y_pred = grid_search.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    
    # Guardar o melhor modelo encontrado
    model_filename = f"trained_model_{model_config['name']}.joblib"
    dump(grid_search.best_estimator_, model_filename)
    print(f"Modelo {model_config['name']} guardado como {model_filename}")
    
    results.append({
        'name': model_config['name'],
        'best_params': grid_search.best_params_,
        'cv_r2': grid_search.best_score_,
        'test_r2': r2_test,
        'model': grid_search
    })

# Comparação final
print("\n" + "="*50)
print("COMPARAÇÃO FINAL")
print("="*50)

for result in results:
    print(f"\n{result['name']}:")
    print(f"  CV R2: {result['cv_r2']:.4f}")
    print(f"  Test R2: {result['test_r2']:.4f}")

best_result = max(results, key=lambda x: x['test_r2'])
print(f"\n Melhor modelo: {best_result['name']} (Test R2 = {best_result['test_r2']:.4f})")


# NOTA: O random forest teve melhor desempenho no test_simul, mas não poderá ser usado na competição, pois não foi abordado nas aulas.
# %%
