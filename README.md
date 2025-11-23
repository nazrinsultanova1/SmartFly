# ✈️ SmartFly - Flight Price Predictor

A web application for predicting airline ticket prices using Machine Learning (Random Forest algorithm).

## 🎯 Features
- Price prediction based on route, date, airline, and other parameters
- User-friendly web interface
- Real-time predictions

## 🛠️ Technologies
- **Python** - Core programming language
- **Flask** - Web framework
- **Random Forest** - Machine Learning algorithm
- **Scikit-learn** - ML library
- **Pandas** - Data processing
- **HTML/CSS** - Frontend

## 📊 About the Model
- **Algorithm**: Random Forest Regressor
- **Model Size**: 472 MB
- **Features**: route, departure date, airline, service class

## 🚀 Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/nazrinsultanova1/SmartFly.git
cd SmartFly
```

### 2. Install dependencies
```bash
pip install flask pandas scikit-learn numpy
```

### 3. Download trained models
Models are stored separately due to large file size (472 MB total).

📥 **[Download all models from Google Drive](https://drive.google.com/drive/folders/1GDkFQlbJY7krBDZPhQy_hdoPj25-NaqC?usp=drive_link)**

Create a `models/` folder and place the downloaded files there:
```
SmartFly/
└── models/              ← create this folder
    ├── random_forest_model.pkl
    ├── label_encoders.pkl
    └── features.pkl
```

### 4. Run the application
```bash
python app.py
```

Open your browser: `http://localhost:5000`

## 📁 Project Structure
```
SmartFly/
├── app.py                      # Main Flask application
├── models/                     # ML models (download separately)
│   ├── random_forest_model.pkl # Trained Random Forest model
│   ├── label_encoders.pkl      # Label encoders for categorical features
│   └── features.pkl            # Feature list
├── templates/                  # HTML templates
│   ├── index.html             # Main page
│   └── predict.html           # Results page
└── static/                     # Static files
    └── style.css              # Styling
```

## 📈 Model Performance
- **Algorithm**: Random Forest Regressor
- **Training Data**: Large dataset of flight records
- **Model Size**: 472 MB of trained parameters

## 💡 How It Works
1. User inputs flight details (route, date, airline, etc.)
2. Model processes the input using trained Random Forest algorithm
3. System returns predicted price range
4. Results displayed in user-friendly interface

## 🖼️ Screenshots
<img width="1284" height="614" alt="image" src="https://github.com/user-attachments/assets/52d1f7c5-6167-4bfc-b75c-0c1844e868f3" />
<img width="1282" height="620" alt="image" src="https://github.com/user-attachments/assets/07f9183e-ae4d-4cdf-8221-ff884e8344d6" />
<img width="1288" height="613" alt="image" src="https://github.com/user-attachments/assets/bdd9b6da-c46e-4b76-bb2d-032e777992b4" />
<img width="1272" height="612" alt="image" src="https://github.com/user-attachments/assets/43d968b0-0f6d-4598-88bf-322677fd7533" />
<img width="1286" height="612" alt="image" src="https://github.com/user-attachments/assets/5df29463-11f1-4f89-b1ee-50b198e9fe21" />
<img width="1280" height="610" alt="image" src="https://github.com/user-attachments/assets/0b7db189-a6cb-46fc-860d-45a6fd4461f1" />
<img width="1282" height="611" alt="image" src="https://github.com/user-attachments/assets/2d8b1183-6be8-4245-b68e-7398298c0779" />







## 👤 Author
**Nəzrin Sultanova**

[LinkedIn](https://www.linkedin.com/in/nəzrin-sultanova-41b691368/) | [GitHub](https://github.com/nazrinsultanova1)

## 📝 Note
Models are not included in the repository due to GitHub file size limitations. Please download them from the Google Drive link above.

## 🔒 License
This project was created for educational purposes.
```
