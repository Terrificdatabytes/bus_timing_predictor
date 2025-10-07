# app.py - Flask Backend
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np

# --- VERCEL SIZE FIX: Optimized Scikit-Learn ---
# Requires 'scikit-learn-intelex' in requirements.txt
from sklearnex import patch_sklearn
patch_sklearn()
# Standard scikit-learn imports now use the optimized, smaller binaries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Matplotlib and plotting functions REMOVED to maintain < 250MB size limit.
# Visualization will be handled by the client (index.html).

import io
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store model and data
model_data = {
    'model': None,
    'scaler': None,
    'predictions': None,
    'actual': None,
    'metrics': None,
    # FIXED: Now includes the required fourth feature
    'feature_names': ['hour', 'weekday', 'scheduled_arrival', 'passenger_count'] 
}

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    """Handles CSV file upload and stores it for training."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate columns
        required_cols = ['timestamp', 'actual_arrival', 'scheduled_arrival', 'passenger_count']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            return jsonify({'error': f"CSV must contain: {', '.join(missing_cols)}"}), 400
        
        # Store in session (for simplicity, using global variable)
        app.config['DATA'] = df
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns)
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload Error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Handles model training based on user configuration."""
    try:
        config = request.json
        model_type = config.get('model_type', 'linear')
        train_split = config.get('train_split', 80) / 100
        
        df = app.config.get('DATA')
        if df is None:
            return jsonify({'error': 'No data uploaded. Please upload a CSV first.'}), 400
        
        # Prepare features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek
        
        # Select features and remove NaN values
        cols_to_use = model_data['feature_names'] + ['actual_arrival']
        df_clean = df[cols_to_use].dropna()
        
        # FIXED: X now contains all four features
        X = df_clean[model_data['feature_names']].values
        y = df_clean['actual_arrival'].values
        
        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        else:
            # Polynomial features (for "dense" equivalent)
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store model and results
        model_data['model'] = model
        model_data['scaler'] = scaler
        model_data['predictions'] = y_pred.tolist()
        model_data['actual'] = y_test.tolist()
        model_data['metrics'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        # Return metrics and raw data for client-side plotting
        return jsonify({
            'success': True,
            'metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 3)
            },
            'actual': model_data['actual'],
            'predictions': model_data['predictions']
        })
    
    except Exception as e:
        # Log the full error to the console for debugging
        print(f"Error during training: {e}")
        return jsonify({'error': f'Training Error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handles real-time prediction based on user input features."""
    try:
        if model_data['model'] is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.json
        hour = float(data.get('hour'))
        weekday = float(data.get('weekday'))
        scheduled = float(data.get('scheduled'))
        # FIXED: Retrieve the fourth feature
        passenger_count = float(data.get('passenger_count')) 
        
        # Prepare input: MUST match the feature order: [hour, weekday, scheduled_arrival, passenger_count]
        X_input = np.array([[hour, weekday, scheduled, passenger_count]])
        X_input_scaled = model_data['scaler'].transform(X_input)
        
        # Predict
        prediction = model_data['model'].predict(X_input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction Error: {str(e)}'}), 500

@app.route('/download_csv')
def download_csv():
    """Allows users to download a CSV file of the test data results."""
    try:
        if model_data['predictions'] is None or model_data['actual'] is None:
            return jsonify({'error': 'No predictions available'}), 400
        
        # Create DataFrame
        df = pd.DataFrame({
            'Actual': model_data['actual'],
            'Predicted': model_data['predictions'],
            'Residual': np.array(model_data['predictions']) - np.array(model_data['actual'])
        })
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='bus_predictions_test_results.csv'
        )
    
    except Exception as e:
        print(f"Error during CSV download: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
