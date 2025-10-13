import fastf1 as ff1
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import os
import plotly.express as px
import difflib
import requests
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Cache
# -----------------------
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# -----------------------
# Circuit features
# -----------------------
CIRCUIT_FEATURES = {
    "Bahrain": {"Length": 5.412, "Corners": 15, "OvertakingDifficulty": 0.25},
    "Jeddah": {"Length": 6.174, "Corners": 27, "OvertakingDifficulty": 0.40},
    "Melbourne": {"Length": 5.278, "Corners": 14, "OvertakingDifficulty": 0.35},
    "Suzuka": {"Length": 5.807, "Corners": 18, "OvertakingDifficulty": 0.35},
    "Shanghai": {"Length": 5.451, "Corners": 16, "OvertakingDifficulty": 0.30},
    "Miami": {"Length": 5.412, "Corners": 19, "OvertakingDifficulty": 0.35},
    "Imola": {"Length": 4.909, "Corners": 19, "OvertakingDifficulty": 0.50},
    "Monaco": {"Length": 3.337, "Corners": 19, "OvertakingDifficulty": 0.90},
    "Montreal": {"Length": 4.361, "Corners": 14, "OvertakingDifficulty": 0.35},
    "Barcelona": {"Length": 4.657, "Corners": 14, "OvertakingDifficulty": 0.45},
    "Red Bull Ring": {"Length": 4.318, "Corners": 10, "OvertakingDifficulty": 0.25},
    "Silverstone": {"Length": 5.891, "Corners": 18, "OvertakingDifficulty": 0.30},
    "Hungaroring": {"Length": 4.381, "Corners": 14, "OvertakingDifficulty": 0.60},
    "Spa": {"Length": 7.004, "Corners": 19, "OvertakingDifficulty": 0.30},
    "Zandvoort": {"Length": 4.259, "Corners": 14, "OvertakingDifficulty": 0.55},
    "Monza": {"Length": 5.793, "Corners": 11, "OvertakingDifficulty": 0.20},
    "Baku": {"Length": 6.003, "Corners": 20, "OvertakingDifficulty": 0.45},
    "Singapore": {"Length": 4.940, "Corners": 19, "OvertakingDifficulty": 0.75},
    "Austin": {"Length": 5.513, "Corners": 20, "OvertakingDifficulty": 0.40},
    "Mexico City": {"Length": 4.304, "Corners": 17, "OvertakingDifficulty": 0.35},
    "Sao Paulo": {"Length": 4.309, "Corners": 15, "OvertakingDifficulty": 0.25},
    "Las Vegas": {"Length": 6.201, "Corners": 17, "OvertakingDifficulty": 0.30},
    "Lusail": {"Length": 5.419, "Corners": 16, "OvertakingDifficulty": 0.35},
    "Abu Dhabi": {"Length": 5.281, "Corners": 16, "OvertakingDifficulty": 0.40}
}

# -----------------------
# Helper: get GP name
# -----------------------
def resolve_gp_input(year, gp_name=None, round_num=None):
    schedule = ff1.get_event_schedule(year)
    if gp_name:
        gps = schedule['EventName'].tolist()
        matches = [g for g in gps if gp_name.lower() in g.lower()]
        if matches:
            return matches[0]
        matches = difflib.get_close_matches(gp_name, gps, n=1, cutoff=0.3)
        if matches:
            return matches[0]
        raise ValueError(f"No GP found matching '{gp_name}' in {year}.")
    if round_num:
        return schedule.iloc[round_num-1]['EventName']
    raise ValueError("Must provide either gp_name or round_num")

# -----------------------
# Data fetching
# -----------------------
def fetch_race_data(year, gp_name):
    try:
        s = ff1.get_session(year, gp_name, 'R')
        s.load(laps=True)
        laps = s.laps
        df = laps[['Driver','LapNumber','Position','Stint','Compound','TyreLife','LapTime','PitInTime']].copy()
        df['LapTime'] = df['LapTime'].dt.total_seconds()
        df['PitStops'] = df['PitInTime'].notnull().astype(int)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_qualifying_data(year, gp_name):
    try:
        s = ff1.get_session(year, gp_name, 'Q')
        s.load(laps=True)
        q = s.laps[['Driver','LapTime']].copy()
        q['LapTime'] = q['LapTime'].dt.total_seconds()
        q['GridPosition'] = q.groupby('Driver')['LapTime'].transform('min').rank(method='first')
        return q.groupby('Driver').agg({'GridPosition':'first','LapTime':'min'}).reset_index()
    except Exception:
        return pd.DataFrame()

def fetch_championship_points(year):
    driver_points, constructor_points = {}, {}
    try:
        r = requests.get(f"https://ergast.com/api/f1/{year}/driverStandings.json", timeout=5)
        r.raise_for_status()
        standings = r.json()['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        for item in standings:
            code = item['Driver'].get('code', item['Driver']['driverId'][:3].upper())
            pts = float(item['points'])
            driver_points[code] = pts
            constructor = item['Constructors'][0]['name']
            constructor_points[constructor] = constructor_points.get(constructor, 0) + pts
    except:
        pass
    return driver_points, constructor_points

# -----------------------
# Features
# -----------------------
def compute_driver_historical_averages(historical_dfs):
    hist = []
    for i, df in enumerate(historical_dfs):
        if df.empty: continue
        w = [0.6,0.3,0.1][i] if i < 3 else 0.1
        g = df.groupby('Driver').agg({'LapTime':'mean','Stint':'mean','PitStops':'sum'}).reset_index()
        g[['LapTime','Stint','PitStops']] *= w
        hist.append(g)
    if not hist:
        return pd.DataFrame(columns=['Driver','HistAvgLapTime','HistStints','HistPitStops'])
    c = pd.concat(hist).groupby('Driver').sum().reset_index()
    c.rename(columns={'LapTime':'HistAvgLapTime','Stint':'HistStints','PitStops':'HistPitStops'}, inplace=True)
    return c

def prepare_dataset(qual_df, hist_df, driver_points, constructor_points, gp_name):
    df = pd.merge(qual_df, hist_df, on='Driver', how='left')
    for c in ['HistAvgLapTime','HistStints','HistPitStops']:
        df[c] = df[c].fillna(df[c].mean() if df[c].notnull().any() else 0)
    df['DriverPoints'] = df['Driver'].map(driver_points).fillna(0)
    df['ConstructorPoints'] = np.mean(list(constructor_points.values())) if constructor_points else 0
    circuit = next((k for k in CIRCUIT_FEATURES if k.lower() in gp_name.lower()), "Monza")
    for k,v in CIRCUIT_FEATURES[circuit].items():
        df[k] = v
    return df

# -----------------------
# Neural net model
# -----------------------
class F1PredictorNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

# -----------------------
# Train models
# -----------------------
def train_xgb(df):
    features = ['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty']
    df['FinishPosition'] = df['HistAvgLapTime'].rank(method='first') + df['GridPosition']*0.5
    df = df.replace([np.inf,-np.inf],np.nan).dropna(subset=features+['FinishPosition'])
    X, y = df[features], df['FinishPosition']
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    print(f"? XGBoost trained. MAE={mean_absolute_error(y, model.predict(X)):.3f}")
    return model, df, features

def train_nn(df, features, epochs=300):
    df = df.replace([np.inf,-np.inf],np.nan).dropna(subset=features)
    X, y = df[features].values, df['FinishPosition'].values
    scaler = StandardScaler(); X = scaler.fit_transform(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = F1PredictorNN(X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    for _ in range(epochs):
        opt.zero_grad(); out = model(X_t); loss = loss_fn(out, y_t); loss.backward(); opt.step()
    print(f"? Neural Net trained. MAE={mean_absolute_error(y, model(X_t).detach().cpu().numpy()):.3f}")
    return model, scaler

# -----------------------
# Predict
# -----------------------
def predict(df, xgb_model, nn_model, scaler, features, alpha=0.5):
    X = df[features].values
    X = scaler.transform(X)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad(): nn_pred = nn_model(X_t).squeeze().numpy()
    xgb_pred = xgb_model.predict(df[features])
    combined = alpha * nn_pred + (1 - alpha) * xgb_pred
    df['PredXGB'], df['PredNN'], df['PredHybrid'] = xgb_pred, nn_pred, combined
    return df

# -----------------------
# Save predictions
# -----------------------
def save_predictions(df, year, gp_name):
    os.makedirs("predictions/csv", exist_ok=True)
    os.makedirs("predictions/html", exist_ok=True)
    csv_path = f"predictions/csv/{year}_predicted_{gp_name.replace(' ','_')}.csv"
    df[['Driver','PredXGB','PredNN','PredHybrid']].to_csv(csv_path,index=False)
    fig = px.bar(df.sort_values('PredHybrid'), x='Driver', y='PredHybrid', title=f'{gp_name} Prediction (Hybrid)')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    html_path = f"predictions/html/{year}_predicted_{gp_name.replace(' ','_')}.html"
    fig.write_html(html_path)
    print(f"?? Saved {csv_path}\n?? Saved {html_path}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument("--round_num", type=int, required=True)
    parser.add_argument("--gp")
    parser.add_argument("--year", type=int, default=pd.Timestamp.now().year)
    args = parser.parse_args()

    gp_name = resolve_gp_input(args.year, gp_name=args.gp, round_num=(args.round_num+1))
    print(f"?? Predicting {gp_name} ({args.year})")

    # Load data
    hist = [fetch_race_data(y, gp_name) for y in [2022,2023,2024]]
    qual = fetch_qualifying_data(args.year, gp_name)
    dp, cp = fetch_championship_points(args.year)
    hist_avg = compute_driver_historical_averages(hist)
    train_df = prepare_dataset(qual, hist_avg, dp, cp, gp_name)
    train_df['FinishPosition'] = train_df['HistAvgLapTime'].rank(method='first') + train_df['GridPosition']*0.5

    # Train both models
    xgb_model, df, features = train_xgb(train_df)
    nn_model, scaler = train_nn(df, features, epochs=200)

    # Combine predictions
    df = predict(df, xgb_model, nn_model, scaler, features, alpha=0.6)
    save_predictions(df, args.year, gp_name)

