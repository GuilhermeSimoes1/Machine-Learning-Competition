#Lets get the dataset train_FuelConsumption.csv from the UCI Machine Learning Repository
# %%
import pandas as pd

# Load the dataset
url = "train_FuelConsumption.csv"

data = pd.read_csv(url)

# lets get the dataset of test_simul_FuelConsumption.csv from the UCI Machine Learning Repository

test_url = "test_simul_FuelConsumption.csv"

test_data = pd.read_csv(test_url)

# Lets do a simple linear regression model to predict the CO2 Emissions based on Engine Size and Fuel Consumption

from sklearn.model_selection import train_test_split

# Define the features and target variable
X = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]

y = data['CO2EMISSIONS']

# i dont need to split the data because i have a separate test dataset

# Create and train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Make predictions on the test dataset
X_test = test_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]
y_test = test_data['CO2EMISSIONS']
y_pred = model.predict(X_test)

# Evaluate the model only r2 score
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# %%
