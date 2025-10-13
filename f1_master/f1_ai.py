import os
import argparse
import pandas as pd
import numpy as np
import fastf1 as ff1
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------------------------------
# Cache Setup
# ----------------------------------------------------
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# ----------------------------------------------------
# Data Fetching
# ----------------------------------------------------
def fetch_race_data(year, gp_name):
    """Fetch race data with driver numbers."""
    try:
        session = ff1.get_session(year, gp_name, 'R')
        session.load(laps=True)
        laps = session.laps
        df = laps[['DriverNumber', 'LapNumber', 'LapTime', 'Stint', 'Compound', 'PitInTime']].copy()
        df['LapTime'] = df['LapTime'].dt.total_seconds()
        df['PitStops'] = df['PitInTime'].notnull().astype(int)
        grouped = df.groupby('DriverNumber').agg({
            'LapTime': 'mean',
            'Stint': 'mean',
            'PitStops': 'sum'
        }).reset_index()
        return grouped
    except Exception as e:
        print(f"?? Could not fetch race data for {gp_name} ({year}): {e}")
        return pd.DataFrame()

def fetch_qualifying_data(year, gp_name):
    """Fetch qualifying data."""
    try:
        session = ff1.get_session(year, gp_name, 'Q')
        session.load(laps=True)
        q = session.laps[['DriverNumber', 'LapTime']].copy()
        q['LapTime'] = q['LapTime'].dt.total_seconds()
        q['GridPosition'] = q.groupby('DriverNumber')['LapTime'].transform('min').rank(method='first')
        return q.groupby('DriverNumber').agg({'GridPosition': 'first', 'LapTime': 'min'}).reset_index()
    except Exception as e:
        print(f"?? Could not fetch qualifying data for {gp_name} ({year}): {e}")
        return pd.DataFrame()

def prepare_dataset(qual_df, race_df):
    """Combine qualifying and race history features."""
    df = pd.merge(qual_df, race_df, on="DriverNumber", how="left")
    df['HistAvgLapTime'] = df['LapTime_y']
    df['LapTime_y'] = df['LapTime_y'].fillna(df['LapTime_y'].mean())
    df['HistPitStops'] = df['PitStops'].fillna(1)
    df['HistStints'] = df['Stint'].fillna(1)
    df = df.rename(columns={'LapTime_x': 'QualLapTime'})
    return df

# ----------------------------------------------------
# Models
# ----------------------------------------------------
def train_xgb(df):
    """Train XGBoost model safely."""
    features = ['GridPosition', 'HistAvgLapTime', 'HistPitStops', 'HistStints']
    df = df.copy()

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(df[features].mean())

    df['FinishPosition'] = (
        df['HistAvgLapTime'].rank(method='first', na_option='keep') + df['GridPosition'] * 0.5
    )
    df['FinishPosition'] = df['FinishPosition'].fillna(df['FinishPosition'].mean())

    X = df[features]
    y = df['FinishPosition']

    if y.isnull().all() or y.nunique() <= 1:
        print("?? Insufficient variation for XGBoost training. Skipping.")
        return None

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1
    )
    model.fit(X, y)
    return model

def train_nn(df):
    """Train Neural Network safely."""
    features = ['GridPosition', 'HistAvgLapTime', 'HistPitStops', 'HistStints']
    df = df.copy()

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(df[features].mean())

    df['FinishPosition'] = (
        df['HistAvgLapTime'].rank(method='first', na_option='keep') + df['GridPosition'] * 0.5
    )
    df['FinishPosition'] = df['FinishPosition'].fillna(df['FinishPosition'].mean())

    X = df[features].values
    y = df['FinishPosition'].values

    if len(X) == 0 or np.isnan(y).any():
        print("?? Skipping NN training: invalid or empty data.")
        return None

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X, y, epochs=50, verbose=0)
    return model

# ----------------------------------------------------
# Prediction Logic
# ----------------------------------------------------
def make_predictions(year, gp_name):
    print(f"?? Generating prediction for {gp_name} ({year})")

    race_df = fetch_race_data(year - 1, gp_name)
    qual_df = fetch_qualifying_data(year, gp_name)
    df = prepare_dataset(qual_df, race_df)

    if df.empty:
        print("?? Not enough data to make predictions.")
        return

    xgb_model = train_xgb(df)
    nn_model = train_nn(df)

    features = ['GridPosition', 'HistAvgLapTime', 'HistPitStops', 'HistStints']
    df['xgb_pred'] = xgb_model.predict(df[features]) if xgb_model else np.zeros(len(df))
    df['nn_pred'] = nn_model.predict(df[features]).flatten() if nn_model else np.zeros(len(df))
    df['PredictedPosition'] = (df['xgb_pred'] + df['nn_pred']) / 2

    output_dir = os.path.join(os.getcwd(), "predictions", "csv")
    os.makedirs(output_dir, exist_ok=True)
    safe_gp_name = gp_name.replace(" ", "_")
    out_file = os.path.join(output_dir, f"{year}_predicted_{safe_gp_name}.csv")
    df.to_csv(out_file, index=False)
    print(f"? Predictions saved: {out_file}")

# ----------------------------------------------------
# Evaluation Logic
# ----------------------------------------------------
def evaluate_accuracy(year, gp_name):
    safe_gp_name = gp_name.replace(" ", "_")
    pred_path = f"predictions/csv/{year}_predicted_{safe_gp_name}.csv"
    result_path = f"actual_results/{year}_{safe_gp_name}_results.csv"

    if not os.path.exists(pred_path) or not os.path.exists(result_path):
        print("? Missing prediction or result file.")
        return None

    preds = pd.read_csv(pred_path)
    actual = pd.read_csv(result_path)

    # Detect driver columns automatically
    pred_driver_col = next((c for c in preds.columns if "driver" in c.lower()), None)
    actual_driver_col = next((c for c in actual.columns if "driver" in c.lower()), None)

    if not pred_driver_col or not actual_driver_col:
        print(f"? Could not find driver columns.\nPreds: {preds.columns.tolist()}\nActual: {actual.columns.tolist()}")
        return None

    preds[pred_driver_col] = preds[pred_driver_col].astype(str).str.strip().str.upper()
    actual[actual_driver_col] = actual[actual_driver_col].astype(str).str.strip().str.upper()

    # Try to merge by driver names or numbers
    merged = preds.merge(actual, left_on=pred_driver_col, right_on=actual_driver_col, suffixes=("_pred", "_actual"))

    if merged.empty:
        print("? No matching drivers between predictions and results!")
        print("Predictions:", preds[pred_driver_col].tolist()[:10])
        print("Results:", actual[actual_driver_col].tolist()[:10])
        return None

    # Locate finish position column
    finish_col = next(
        (col for col in ["FinishPosition_actual", "FinishPosition", "Position_actual", "Position"]
         if col in merged.columns),
        None,
    )

    if not finish_col:
        print(f"? No finish position column found. Columns: {merged.columns.tolist()}")
        return None

    # Compute metrics
    mae = mean_absolute_error(merged[finish_col], merged["PredictedPosition"])
    acc = (1 - mae / 20) * 100

    top3 = np.mean(
        merged.nsmallest(3, "PredictedPosition")[pred_driver_col].isin(
            merged.nsmallest(3, finish_col)[actual_driver_col]
        )
    ) * 100
    top10 = np.mean(
        merged.nsmallest(10, "PredictedPosition")[pred_driver_col].isin(
            merged.nsmallest(10, finish_col)[actual_driver_col]
        )
    ) * 100

    print(f"\n?? Accuracy Report for {gp_name} ({year})")
    print(f"MAE: {mae:.2f}")
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Top-3 Accuracy: {top3:.2f}%")
    print(f"Top-10 Accuracy: {top10:.2f}%")

# ----------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["predict", "update", "evaluate"])
    parser.add_argument("--year", type=int, default=pd.Timestamp.now().year)
    parser.add_argument("--gp_name", type=str)
    parser.add_argument("--round_num", type=int, required=True)
    args = parser.parse_args()

    if args.mode == "predict":
        make_predictions(args.year, args.gp_name)
    elif args.mode == "evaluate":
        evaluate_accuracy(args.year, args.gp_name)
    elif args.mode == "update":
        print("?? Model updating feature coming soon.")

if __name__ == "__main__":
    main()

