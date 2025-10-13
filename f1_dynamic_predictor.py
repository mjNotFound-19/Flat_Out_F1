import fastf1 as ff1
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import joblib
import plotly.graph_objects as go

# Setup cache and folders
os.makedirs("f1_cache", exist_ok=True)
ff1.Cache.enable_cache("f1_cache")
os.makedirs("Datasets", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# Overtaking difficulty mapping (example)
OVERTAKING = {
    'Monza': 1,
    'Spa-Francorchamps': 2,
    'Silverstone': 2,
    'Singapore': 5,
    'Monaco': 5,
    'Azerbaijan': 3
}

# Helper: find GP round dynamically
def get_gp_round_by_name(year, gp_name):
    try:
        schedule = ff1.get_event_schedule(year)
        schedule['EventName_lower'] = schedule['EventName'].str.lower()
        gp_row = schedule[schedule['EventName_lower'].str.contains(gp_name.lower())]
        if not gp_row.empty:
            return int(gp_row['RoundNumber'].values[0])
        else:
            print(f"No GP found matching '{gp_name}' in {year}")
            return None
    except Exception as e:
        print(f"Failed to get schedule for {year}: {e}")
        return None

# Fetch historical data for a GP in a specific year
def fetch_historical_gp(year, gp_name):
    round_number = get_gp_round_by_name(year, gp_name)
    if round_number is None:
        return pd.DataFrame(), None
    try:
        q = ff1.get_session(year, round_number, 'Q'); q.load()
        r = ff1.get_session(year, round_number, 'R'); r.load(laps=True)

        qres = q.results[['Abbreviation','Position','Q1','Q2','Q3']].copy()
        qres.rename(columns={'Position':'GridPosition'}, inplace=True)
        qres['BestQualiTime'] = qres[['Q1','Q2','Q3']].min(axis=1).dt.total_seconds()

        laps = r.laps
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        race_feat = laps.groupby('Driver').agg(
            AvgLapTime=('LapTimeSec','mean'),
            PitStops=('PitInTime', lambda x:x.notnull().sum()),
            Stints=('Stint','max')
        ).reset_index()

        df = pd.merge(qres[['Abbreviation','GridPosition','BestQualiTime']], race_feat, left_on='Abbreviation', right_on='Driver')
        df['FinalPosition'] = r.results.set_index('Abbreviation').loc[df['Abbreviation'],'Position'].values
        df['Year'] = year
        df['Round'] = round_number
        df['GP'] = gp_name
        df['OvertakingDifficulty'] = OVERTAKING.get(gp_name.split()[0],3)
        return df, gp_name
    except Exception as e:
        print(f"Skipping {year} {gp_name}: {e}")
        return pd.DataFrame(), None

# Build training dataset from past years
def build_training_data(years, gp_name):
    dfs = []
    for year in years:
        df, gp = fetch_historical_gp(year, gp_name)
        if not df.empty:
            dfs.append(df)
    if dfs:
        dataset = pd.concat(dfs, ignore_index=True)
        dataset.to_csv(f"Datasets/f1_training_{gp_name.replace(' ','_')}.csv", index=False)
        return dataset
    return pd.DataFrame()

# Compute driver historical averages
def compute_driver_historical_averages(dataset):
    return dataset.groupby('Abbreviation').agg(
        HistAvgLapTime=('AvgLapTime','mean'),
        HistPitStops=('PitStops','mean'),
        HistStints=('Stints','mean')
    ).reset_index()

# Train LightGBM model
def train_model(df):
    features = ['GridPosition','BestQualiTime','HistAvgLapTime','HistPitStops','HistStints','OvertakingDifficulty']
    X = df[features].fillna(0)
    y = df['FinalPosition']

    if not HAS_LGB:
        raise ImportError("LightGBM not installed")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    models = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            {'objective':'regression','metric':'mae','verbosity':-1},
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )
        models.append(model)
        maes.append(mean_absolute_error(y_val, model.predict(X_val)))

    print(f"CV MAE: {np.mean(maes):.3f}")
    joblib.dump(models, f"Models/f1_model_{df['GP'].iloc[0].replace(' ','_')}.pkl")
    return models

# Predict upcoming race
def predict_upcoming_race(year, round_number, gp_name, models, hist_avgs):
    try:
        q = ff1.get_session(year, round_number, 'Q'); q.load()
        qres = q.results[['Abbreviation','Position','Q1','Q2','Q3']].copy()
        qres.rename(columns={'Position':'GridPosition'}, inplace=True)
        qres['BestQualiTime'] = qres[['Q1','Q2','Q3']].min(axis=1).dt.total_seconds()

        df_pred = pd.merge(qres, hist_avgs, on='Abbreviation', how='left')

        # Fix FutureWarning by using direct assignment
        df_pred['HistAvgLapTime'] = df_pred['HistAvgLapTime'].fillna(df_pred['HistAvgLapTime'].mean())
        df_pred['HistPitStops'] = df_pred['HistPitStops'].fillna(df_pred['HistPitStops'].mean())
        df_pred['HistStints'] = df_pred['HistStints'].fillna(df_pred['HistStints'].mean())

        df_pred['OvertakingDifficulty'] = OVERTAKING.get(gp_name.split()[0],3)

        features = ['GridPosition','BestQualiTime','HistAvgLapTime','HistPitStops','HistStints','OvertakingDifficulty']
        X_pred = df_pred[features].fillna(0)
        preds = np.zeros(len(X_pred))
        for m in models: preds += m.predict(X_pred)
        preds /= len(models)

        df_pred['PredictedFinish'] = preds
        df_sorted = df_pred.sort_values('PredictedFinish').reset_index(drop=True)
        df_sorted['PredictedRank'] = np.arange(1,len(df_sorted)+1)

        # Ensure all columns are JSON serializable for Plotly
        df_sorted = df_sorted.copy()
        for col in df_sorted.columns:
            if pd.api.types.is_timedelta64_dtype(df_sorted[col]):
                df_sorted[col] = df_sorted[col].dt.total_seconds()
            elif pd.api.types.is_datetime64_any_dtype(df_sorted[col]):
                df_sorted[col] = df_sorted[col].astype(str)

        # Save CSV
        out_csv = f"Datasets/predicted_{year}_R{round_number}_{gp_name.replace(' ','_')}_upcoming.csv"
        df_sorted.to_csv(out_csv,index=False)
        print(f"Predictions saved: {out_csv}")

        # Save interactive HTML
        out_html = out_csv.replace('.csv','.html')
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df_sorted.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[df_sorted[col] for col in df_sorted.columns], fill_color='lavender', align='left')
        )])
        fig.write_html(out_html, include_plotlyjs='cdn')
        print(f"Interactive HTML saved: {out_html}")

        return df_sorted
    except Exception as e:
        print(f"Could not fetch qualifying for upcoming race: {e}")
        return pd.DataFrame()

def prepare_dataset_for_training(dataset):
    """Merge historical averages into dataset and fill missing values."""
    # Compute historical averages
    hist_avgs = compute_driver_historical_averages(dataset)

    # Merge historical averages
    dataset = pd.merge(dataset, hist_avgs, left_on='Abbreviation', right_on='Abbreviation', how='left')

    # Fill missing values
    dataset['HistAvgLapTime'] = dataset['HistAvgLapTime'].fillna(dataset['HistAvgLapTime'].mean())
    dataset['HistPitStops'] = dataset['HistPitStops'].fillna(dataset['HistPitStops'].mean())
    dataset['HistStints'] = dataset['HistStints'].fillna(dataset['HistStints'].mean())

    return dataset, hist_avgs


# =====================
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="F1 Race Predictor")
    parser.add_argument('--gp', type=str, help='Target GP name (e.g., Monza)')
    parser.add_argument('--year', type=int, help='Target year (default=most recent)')
    parser.add_argument('--round_num', type=int, help='Specific round number')
    parser.add_argument('--seasons', nargs='+', type=int, default=[2021,2022,2023,2024], help='Historical years')
    args = parser.parse_args()

    if not args.gp and not args.year and not args.round_num:
        print("Please provide at least one argument: --gp, --year, or --round_num")
        exit()

    # Determine year
    year = args.year if args.year else pd.Timestamp.now().year

    # Determine GP and round dynamically
    if args.gp:
        gp_name = args.gp
        round_number = get_gp_round_by_name(year, gp_name) if not args.round_num else args.round_num
    elif args.round_num:
        round_number = args.round_num
        schedule = ff1.get_event_schedule(year)
        gp_name = schedule[schedule['RoundNumber']==round_number]['EventName'].values[0]

    # Build training dataset
    dataset = build_training_data(args.seasons, gp_name)
    if dataset.empty:
        print(f"No historical data found for GP {gp_name}")
        exit()

    # Prepare dataset for training (merge historical averages and fill missing values)
    dataset, hist_avgs = prepare_dataset_for_training(dataset)

    # Train models
    models = train_model(dataset)

    # Predict upcoming race
    predict_upcoming_race(year, round_number, gp_name, models, hist_avgs)
