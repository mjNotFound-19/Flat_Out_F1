# 🏎️ Flat_Out_F1 
**Formula 1 Race Prediction & Evaluation Framework**

This project, **Flat_Out_F1**, is a Python-based pipeline designed to **predict**, **evaluate**, and **incrementally update** Formula 1 race outcomes using **FastF1**, **XGBoost**, and **Neural Networks**. It leverages historical qualifying and race data to forecast finishing positions and iteratively improve its models after each race weekend.

---

## 📁 Project Structure

```
.
├── f1_ai_v2.py              # Core script (main entrypoint)
├── f1_cache/                # FastF1 cache directory (auto-generated)
├── predictions/
│   └── csv/                 # Stores model predictions per Grand Prix
├── actual_results/          # Store actual F1 race results (CSV)
├── models/
│   ├── f1_xgb.json          # Trained XGBoost model
│   └── f1_nn.keras          # Trained Neural Network
└── README.md                # (this file)
```

---

## ⚙️ Features

### 🔹 **Data Handling**
- Automatically fetches **race** and **qualifying** data from `fastf1`.
- Cleans, aggregates, and merges **driver lap times, stints, and pit stops**.
- Caches requests to `f1_cache` for faster repeated queries.

### 🔹 **Machine Learning Models**
- **XGBoost Regression** (`xgb.XGBRegressor`)  
  Learns relationships between qualifying position, lap averages, and stints.
- **Neural Network (Keras)**  
  A two-layer dense NN that complements the tree-based model.
- Combines predictions from both models for **ensemble forecasting**.

### 🔹 **Evaluation**
- Compares predicted vs actual race outcomes.
- Computes:
  - Mean Absolute Error (MAE)
  - Overall Accuracy (%)
  - Winner Accuracy
  - Top-3 and Top-10 prediction accuracy
- Outputs a detailed accuracy report for each race.

### 🔹 **Incremental Model Updates**
- Continuously retrains on actual race results to refine future predictions.
- Stores updated models in the `models/` directory.

---

## 🚀 Usage

### 1️⃣ **Prediction Mode**
Generate finishing position predictions for an upcoming race.

```bash
python f1_ai_v2.py --mode predict --year 2025 --round_num 3
```

> The script automatically resolves the Grand Prix name using `fastf1` if not provided.

---

### 2️⃣ **Evaluation Mode**
Compare predictions with actual results after a race.

```bash
python f1_ai_v2.py --mode evaluate --year 2025 --round_num 3
```

Expected folder structure:
```
predictions/csv/2025_predicted_Japan_Grand_Prix.csv
actual_results/2025_Japan_Grand_Prix_results.csv
```

---

### 3️⃣ **Update Mode**
Retrain the models incrementally using the latest race results.

```bash
python f1_ai_v2.py --mode update --year 2025 --round_num 3
```

This will:
- Merge prediction and actual datasets.
- Continue training existing `f1_xgb.json` and `f1_nn.keras`.
- Save updated models.

---

## 📊 Example Output

```
Round 18: Japan Grand Prix (2025)

Generating prediction for Japan Grand Prix (2025)
Predictions saved: /predictions/csv/2025_predicted_Japan_Grand_Prix.csv

Accuracy Report for Japan Grand Prix (2025)
MAE: 61.73
Overall Accuracy: 0.00%
Winner Accuracy: 100.00%
Top-3 Accuracy: 100.00%
Top-10 Accuracy: 90.00%
```

---

## 🧠 Model Inputs

| Feature | Description |
|----------|--------------|
| `GridPosition` | Starting grid rank from qualifying |
| `HistAvgLapTime` | Mean lap time from previous season |
| `HistPitStops` | Total number of pit stops in race |
| `HistStints` | Average stint length |

---

## 💡 Design Highlights

- **Auto GP Resolution** — Fetches Grand Prix name from round number automatically.  
- **Cache Optimization** — Uses FastF1’s caching to avoid redundant downloads.  
- **Safe Training** — Handles NaNs, infinite values, and empty data gracefully.  
- **Cross-Model Ensemble** — Combines XGBoost and Neural Net predictions for balance.  

---

## 🧩 Dependencies

Install dependencies using:

```bash
pip install fastf1 xgboost tensorflow scikit-learn pandas numpy
```

---

## 🏁 Author

**Manas Jha**  
AI & Data Science Researcher | Purdue University  
💻 [GitHub](https://github.com/mjNotFound-19) | 🧠 [LinkedIn](https://www.linkedin.com/in/manas-jha-853708206)
