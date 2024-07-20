import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df

data = pd.read_csv('concrete_data.csv')

columns_to_clean = ['concrete_compressive_strength', 'water', 'blast_furnace_slag', 'superplasticizer', 'age', 'fine_aggregate ']

for column in columns_to_clean:
    data = remove_outliers(data, column)

X = data.drop(columns=['concrete_compressive_strength']) 
y = data['concrete_compressive_strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBRegressor()
model.fit(X_train_scaled, y_train)

pickle.dump(model, open('cement.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler trained and saved successfully!")
