from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
import time
import math # Добавлен для математических операций

app = Flask(__name__)

def get_css_version():
    css_path = 'static/style.css'
    if os.path.exists(css_path):
        return int(os.path.getmtime(css_path))
    return int(time.time())

try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/label_encoders.pkl', 'rb') as f:
        encodings = pickle.load(f)
    
    with open('models/features.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    
except FileNotFoundError:
    print("❌ Model files not found.")
    model = None
    encodings = {
        'airline': {}, 'source_city': {}, 'departure_time': {}, 
        'stops': {}, 'destination_city': {}, 'class': {}
    }
    feature_columns = []

model_comparison = {
    'RandomForest': {'R2': 0.9834, 'RMSE': 34.60, 'MAE': 13.15},
    'GradientBoosting': {'R2': 0.9570, 'RMSE': 55.77, 'MAE': 33.43},
    'XGBoost': {'R2': 0.9808, 'RMSE': 37.29, 'MAE': 20.38},
    'LightGBM': {'R2': 0.9759, 'RMSE': 41.74, 'MAE': 24.13},
    'AdaBoost': {'R2': 0.9349, 'RMSE': 68.62, 'MAE': 43.02},
    'CatBoost': {'R2': 0.9802, 'RMSE': 37.84, 'MAE': 20.85},
    'LinearRegression': {'R2': 0.9057, 'RMSE': 82.59, 'MAE': 55.18}
}

@app.context_processor
def inject_css_version():
    return {'css_version': get_css_version()}

@app.route('/')
def index():
    return render_template('index.html', models=model_comparison)

@app.route('/predict')
def predict_page():
    today = datetime.now().date()
    max_date = today + timedelta(days=365)
    return render_template('predict.html', 
                          min_date=today.isoformat(),
                          max_date=max_date.isoformat())

# --- НОВАЯ ФУНКЦИЯ КЛАССИФИКАЦИИ ЦЕН ---
def classify_prices(prices, base_price=None):
    if not prices:
        return []

    # Используем все 30 цен для определения порогов
    prices_array = np.array(prices)
    mean_price = np.mean(prices_array)
    std_dev = np.std(prices_array)
    
    # Пороги: Cheap < (Mean - 0.5 * StdDev), Expensive > (Mean + 0.5 * StdDev)
    threshold_cheap = mean_price - 0.5 * std_dev
    threshold_expensive = mean_price + 0.5 * std_dev
    
    categories = []
    for price in prices:
        if price < threshold_cheap:
            categories.append("Cheap")
        elif price > threshold_expensive:
            categories.append("Expensive")
        else:
            categories.append("Average")
    return categories

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        data = request.json
        
        # 1. Сначала рассчитываем все 30 цен без категории
        raw_predictions = []
        current_date = datetime.now()
        
        for days_left in range(1, 31):
            input_data = {}
            for feature in feature_columns:
                if feature in ['airline', 'source_city', 'departure_time', 'stops', 'destination_city', 'class']:
                    # Проверяем наличие ключа в data и наличие значения в энкодере
                    value = data.get(feature)
                    if value in encodings.get(feature, {}).classes_:
                        input_data[feature] = encodings[feature].transform([value])[0]
                    else:
                        input_data[feature] = 0
                elif feature == 'days_left':
                    input_data[feature] = days_left
                else:
                    input_data[feature] = data.get(feature, 0) # Используем .get для безопасности
            
            # Обеспечиваем, что все признаки присутствуют
            features_df = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0)
            
            price_usd = model.predict(features_df)[0]
            price_inr = price_usd * 84
            flight_date = current_date + timedelta(days=days_left)
            
            raw_predictions.append({
                'days_until_flight': days_left,
                'flight_date': flight_date.strftime('%d.%m.%Y'),
                'price_usd': max(20, round(price_usd, 2)),
                'price_inr': max(1680, round(price_inr))
            })
            
        # 2. Получаем список только цен USD
        price_list_usd = [p['price_usd'] for p in raw_predictions]
        
        # 3. Классифицируем цены
        price_categories = classify_prices(price_list_usd)
        
        # 4. Объединяем категории с прогнозами
        final_predictions = []
        for i, pred in enumerate(raw_predictions):
            pred['price_category'] = price_categories[i]
            final_predictions.append(pred)
            
        return jsonify({
            'success': True,
            'predictions': final_predictions,
            'route': f"{data['source_city']} → {data['destination_city']}",
            'flight_class': data['class'],
            'airline': data['airline'],
            'departure_time': data['departure_time']
        })
        
    except Exception as e:
        # Добавляем более детальное логирование ошибки
        print(f"Prediction Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Flight Price Predictor...")
    app.run(debug=True, host='0.0.0.0', port=5000)