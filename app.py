from flask import Flask, jsonify,request
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
winvision = pd.read_csv(r"winvision.csv")
drivers = pd.read_csv(r"drivers.csv")
circuits = pd.read_csv(r"circuits.csv")


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
model=pickle.load(open('winvision_model.pkl','rb'))
def prediction(driver_name, grid, circuit_loc):
    driver = drivers.loc[drivers['Name']==driver_name, 'driverId'].iloc[0]
    circuit = circuits.loc[circuits['location']==circuit_loc, ['circuitId']].iloc[0]

    input_data = winvision[winvision['driverId'] == driver].sort_values(by='date', ascending=False).iloc[0]
    circuit_data = circuits[circuits['location']==circuit_loc].iloc[0]

    features = {
        'driverId': input_data['driverId'],
        'constructorId': input_data['constructorId'],
        'grid': grid,
        'laps': input_data['laps'],
        'circuitId': circuit_data['circuitId'],
        'Constructor Experience': input_data['Constructor Experience'],
        'Driver Experience': input_data['Driver Experience'],
        'age': input_data['age'],
        'driver_wins': input_data['driver_wins'],
        'constructor_wins': input_data['constructor_wins'],
        'prev_position': input_data['prev_position'],
        'Driver Constructor Experience': input_data['Driver Constructor Experience'],
        'DNF Score': input_data['DNF Score']
        
    }
    features = pd.DataFrame([features])
    predi = model.predict(features)
    prob= model.predict_proba(features)
    return predi,prob
@app.route('/')
def hello_world():
    return "hello world"

@app.route('/predict',methods=['POST','GET'])
def predict():
    data = request.json
    selected_drivers = data.get('selectedDrivers')
    circuit_loc = data.get('circuit')
    dnfs = data.get('dnfs', [])

    # Reload the datasets (if necessary)
    winvision = pd.read_csv(r"winvision.csv")
    drivers = pd.read_csv(r"drivers.csv")
    drivers['Name'] = drivers['forename'] + ' ' + drivers['surname']
    circuits = pd.read_csv(r"circuits.csv")

    # Filter out the DNF drivers
    remaining_drivers = [driver for driver in selected_drivers if driver not in dnfs]
    remaining_drivers += ['Dummy Driver'] * (20 - len(remaining_drivers))
    grids = list(range(1, len(remaining_drivers) + 1))

    predictions = []

    # Make predictions for the remaining drivers
    for driver_name, grid in zip(remaining_drivers, grids):
        predi, prob = prediction(driver_name, grid, circuit_loc)
        if predi in [1, 2, 3]:
            probability = np.max(prob)
            predictions.append({
                'Driver Name': driver_name,
                'Grid': grid,
                'Prediction': int(predi[0]),
                'Probability': probability
            })

    # Sort predictions by grid position and probability
    predictions.sort(key=lambda x: (x['Prediction'], -x['Probability']))

    # Adjust positions if there are duplicates
    final_predictions = []
    last_position = 0
    for i, pred in enumerate(predictions):
        if i > 0 and predictions[i]['Prediction'] == predictions[i - 1]['Prediction']:
            last_position += 1
        else:
            last_position = pred['Prediction']
        final_predictions.append({
        'Driver Name': pred['Driver Name'],
        'Grid': pred['Grid'],
        'Prediction': last_position,
        'Probability': pred['Probability']
    })

# Final check to ensure only the top 3 predictions for each position are kept
    filtered_predictions = {}
    for pred in final_predictions:
        pos = pred['Prediction']
        if pos not in filtered_predictions:
            filtered_predictions[pos] = []
        filtered_predictions[pos].append(pred)

# Keep only the top 3 predictions for each position based on probability
    top_predictions = []
    for pos, preds in filtered_predictions.items():
        preds.sort(key=lambda x: -x['Probability'])
        top_predictions.extend(preds[:3])

# Sort the final predictions by position and probability
    top_predictions.sort(key=lambda x: (x['Prediction'], -x['Probability']))

# Format the probabilities as percentages
    formatted_output = [f"Driver Name: {pred['Driver Name']}, Grid: {pred['Grid']}, Prediction: {pred['Prediction']}, Probability: {pred['Probability'] * 100:.2f}%"
    for pred in top_predictions]
    return jsonify(formatted_output)

if __name__ == '__main__':
    app.run(debug=True)