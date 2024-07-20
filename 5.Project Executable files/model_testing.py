import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('concrete_data.csv') 

X = data.drop(columns=['concrete_compressive_strength']) 
y = data['concrete_compressive_strength']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('cement.pkl', 'rb'))

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('model_predictions.csv', index=False)
print("Predictions saved to model_predictions.csv")