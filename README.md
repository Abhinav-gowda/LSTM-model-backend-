# LSTM Model Backend - Ingredient IQ API

A Flask-based REST API that uses an LSTM deep learning model to analyze cosmetic and skincare product ingredients for harmfulness.

## 🚀 Live API

**Base URL:** `https://ingredient-iq-api.onrender.com`

## 📋 Files

- `app.py` - Flask API application
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `lstm_model.keras` - Trained LSTM model (3.2 MB)
- `tokenizer.pkl` - Keras tokenizer for text preprocessing
- `ingredient_effects.csv` - Dataset of known ingredients and their effects

## 🔧 Local Setup

```bash
# Clone the repository
git clone https://github.com/Abhinav-gowda/LSTM-model-backend-.git
cd LSTM-model-backend-

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

The API will be available at `http://localhost:5000`

## 📖 API Endpoints

### Health Check
```http
GET /
```
Returns API status.

### Analyze Ingredients
```http
POST /analyze
Content-Type: application/json

{
  "ingredients": ["Water", "Parabens", "Vitamin E", "Fragrance"]
}
```

**Response:**
```json
{
  "ingredients": [
    {
      "name": "Water",
      "effect": "Hydrating agent, generally safe",
      "score": 2.5,
      "source": "dataset",
      "risk_level": "Safe"
    }
  ],
  "product_summary": {
    "average_score": 3.2,
    "maximum_score": 5.8,
    "final_score": 4.5,
    "classification": "Safe",
    "high_risk_ingredients": []
  }
}
```

### Get Single Ingredient
```http
GET /ingredient/Parabens
```

## 🏗️ Deployment

This API is deployed on **Render** using the configuration in `render.yaml`.

### Deployment Steps:
1. Push all files to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically detect `render.yaml` and deploy

> **Note:** If `lstm_model.keras` is over 100 MB, run:
> ```bash
> git lfs install
> git lfs track "*.keras"
> ```

## 🛠️ Tech Stack

- **Backend:** Flask 3.0.3
- **ML:** TensorFlow 2.16.2
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Deployment:** Render (Free tier)

## 📝 License

MIT License

## 👤 Author

- GitHub: [@Abhinav-gowda](https://github.com/Abhinav-gowda)
