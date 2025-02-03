import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle


data = {
    'Bedrooms': [3, 4, 2, 3, 5],
    'Bathrooms': [2, 3, 1, 2, 3],
    'Square_Feet': [1500, 2000, 1000, 1200, 2500],
    'Location': ['Suburbs', 'Downtown', 'Suburbs', 'Suburbs', 'Downtown'],
    'Price': [4000000, 6000000, 2500000, 3500000, 7500000]
}


df = pd.DataFrame(data)


le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])


with open('location_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)


X = df[['Bedrooms', 'Bathrooms', 'Square_Feet', 'Location']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
