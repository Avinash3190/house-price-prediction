from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('location_encoder.pkl', 'rb') as f:
    location_encoder = pickle.load(f)

def predict_price(bedrooms, bathrooms, square_feet, location):
   
    location_encoded = location_encoder.transform([location])[0]  
    input_data = np.array([[bedrooms, bathrooms, square_feet, location_encoded]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

@app.route('/', methods=['GET', 'POST'])
def house_price_prediction():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            
            bedrooms = int(request.form['Bedroom'])
            bathrooms = int(request.form['Bathrooms'])
            square_feet = float(request.form['square_feet'])
            location = request.form['Location']
            
          
            prediction = predict_price(bedrooms, bathrooms, square_feet, location)
        except Exception as e:
            error = f"Error in prediction: {str(e)}"
            print(f"Error: {str(e)}")  
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
