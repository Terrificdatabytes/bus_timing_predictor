# app.py - Flask Backend
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import base64
from datetime import datetime
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
    # UPDATED: Added 'passenger_count' to the list of features
    'feature_names': ['hour', 'weekday', 'scheduled_arrival', 'passenger_count'] 
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        file_content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content))
        
        # Validate columns
        # 'passenger_count' is now a required column for 4-feature prediction
        required_cols = ['timestamp', 'actual_arrival', 'scheduled_arrival', 'passenger_count']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'CSV must contain: {", ".join(required_cols)}'}), 400
        
        # Store in session (for simplicity, using global variable)
        app.config['DATA'] = df
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        config = request.json
        model_type = config.get('model_type', 'linear')
        train_split = config.get('train_split', 80) / 100
        
        df = app.config.get('DATA')
        if df is None:
            return jsonify({'error': 'No data uploaded'}), 400
        
        # Prepare features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek
        
        # Define the list of features for X
        feature_cols = ['hour', 'weekday', 'scheduled_arrival', 'passenger_count']
        
        # Remove NaN values from required columns
        df_clean = df[feature_cols + ['actual_arrival']].dropna()
        
        # UPDATED: X now includes all four features
        X = df_clean[feature_cols].values
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
            # Polynomial features for "dense" equivalent
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
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
        model_data['predictions'] = y_pred
        model_data['actual'] = y_test
        model_data['metrics'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        # Generate plots
        plot1 = generate_pred_vs_actual_plot(y_test, y_pred)
        # Note: The frontend only uses the actual/predicted data for client-side plotting via Plotly, 
        # but we'll include the base64 plots for consistency if a server-side plot is desired.
        
        return jsonify({
            'success': True,
            'metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 3)
            },
            # Return actual and predictions for client-side Plotly visualization
            'actual': y_test.tolist(), 
            'predictions': y_pred.tolist(),
            
            # The original two plots are kept for completeness, though index.html only uses Plotly now
            'plot_pred_vs_actual': plot1,
            'plot_residuals': 'Plotting is now handled by Plotly on the client.' 
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pred_vs_actual_plot(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#667eea', s=50)
    
    # Ideal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    plt.xlabel('Actual Arrival (min)', fontsize=12)
    plt.ylabel('Predicted Arrival (min)', fontsize=12)
    plt.title('Predicted vs Actual Arrival Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data['model'] is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.json
        hour = float(data['hour'])
        weekday = float(data['weekday'])
        scheduled = float(data['scheduled'])
        # ADDED: Extract the new passenger_count feature
        passenger_count = float(data['passenger_count'])
        
        # Prepare input
        # UPDATED: X_input now contains all four features
        X_input = np.array([[hour, weekday, scheduled, passenger_count]])
        X_input_scaled = model_data['scaler'].transform(X_input)
        
        # Predict
        prediction = model_data['model'].predict(X_input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv')
def download_csv():
    try:
        if model_data['predictions'] is None:
            return jsonify({'error': 'No predictions available'}), 400
        
        # Create DataFrame
        df = pd.DataFrame({
            'Actual': model_data['actual'],
            'Predicted': model_data['predictions'],
            'Residual': model_data['predictions'] - model_data['actual']
        })
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='bus_predictions.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
