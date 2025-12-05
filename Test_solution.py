#%%
from joblib import load
import pandas as pd
from sklearn.metrics import r2_score

print("TESTING SOLUTION")
modeljob_test = load("trained_model_GuilhermeSimoes.joblib")
test = pd.read_csv("test_simul_FuelConsumption.csv")

features = [
    'ENGINESIZE',
    'CYLINDERS',
    'FUELCONSUMPTION_CITY',
    'FUELCONSUMPTION_HWY',
    'FUELCONSUMPTION_COMB',
    'FUELCONSUMPTION_COMB_MPG'
]

X_test = test[features]
y_test = test['CO2EMISSIONS']

y_pred = modeljob_test.predict(X_test)
print("Final score =", r2_score(y_test, y_pred))

# %%
