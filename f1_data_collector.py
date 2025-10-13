import fastf1 as ff1
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import joblib

# Enable FastF1 cache
os.makedirs("f1_cache", exist_ok=True)
ff1.Cache.enable_cache("f1_cache")

# Ensure folders
os.makedirs("Datasets", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# Overtaking difficulty (static lookup, 1=easiest, 5=hardest)
OVERTAKING = {
    'Monza': 1,
    'Spa-Francorchamps': 2,
    'Silverstone': 2,
    'Singapore': 5,
    'Monaco': 5
}

def fetch_qualifying_and_race(year, round_number):
    try:
        q_session = ff1.get_session(year, round_number, 'Q')
        q_session.load()
        r_session = ff1.get_session(year, round_number, 'R')
        r_session.load(laps=True)

        gp_name = r_session.event['EventName']

        # Qualifying: get grid positions and best Q time
        q_results = q_session.results[['Abbreviation','Position','Q1','Q2','Q3']].copy()
        q_results.rename(columns={'Position':'GridPosition'}, inplace=True)
        q_results['BestQualiTime'] = q_results[['Q1','Q2','Q3']].min(axis=1).dt.total_seconds()

        # Race: lap features
        laps = r_session.laps
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        race_features = laps.groupby('Driver').agg(
            AvgLapTime=('LapTimeSec','mean'),
            PitStops=('PitInTime', lambda x: x.notnull().sum()),
            Stints=('Stint','max')
        ).reset_index()

        # Merge Quali + Race
        df = pd.merge(q_results[['Abbreviation','GridPosition','BestQualiTime']],
                      race_features, left_on='Abbreviation', right_on='Driver')
        df['FinalPosition'] = r_session.results.set_index('Abbreviation').loc[df['Abbreviation'],'Position'].values
        df['Year'] = year
        df['Round'] = round_number
        df['GP'] = gp_name
        df['OvertakingDifficulty'] = OVERTAKING.get(gp_name.split()[0],3)

        return df, gp_name

    except Exception as e:
        print(f"Skipping {year} Round {round_number}: {e}")
        return pd.DataFrame(), None


def build_circuit_training_data(seasons, target_gp):
    dfs = []
    for year in seasons:
        schedule = ff1.get_event_schedule(year)
        for rnd in schedule.RoundNumber:
            df, gp_name = fetch_qualifying_and_race(year, rnd)
            if df.empty:
                continue
            if gp_name == target_gp:
                dfs.append(df)
    if dfs:
        dataset = pd.concat(dfs, ignore_index=True)
        dataset.to_csv(f"Datasets/f1_training_{target_gp.replace(' ','_')}.csv", index=False)
        return dataset
    else:
        return pd.DataFrame()


def train_lightgbm(df):
    features = ['GridPosition','BestQualiTime','AvgLapTime','PitStops','Stints','OvertakingDifficulty']
    X = df[features].fillna(0)
    y = df['FinalPosition']

    if not HAS_LGB:
        raise ImportError("LightGBM not installed")

    params = {'objective':'regression','metric':'mae','verbosity':-1}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    maes = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=500, early_stopping_rounds=20, verbose_eval=False)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        models.append(model)

    print(f"CV MAE: {np.mean(maes):.3f}")
    # Save first model (you can ensemble later if you want)
    models[0].save_model(f"Models/f1_model_{df['GP'].iloc[0].replace(' ','_')}.txt")
    return models[0]


def predict_race(year, round_number, model):
    df, gp_name = fetch_qualifying_and_race(year, round_number)
    if df.empty:
        print("No data for target race.")
        return

    features = ['GridPosition','BestQualiTime','AvgLapTime','PitStops','Stints','OvertakingDifficulty']
    X_pred = df[features].fillna(0)
    df['PredictedFinish'] = model.predict(X_pred)
    df_sorted = df.sort_values('PredictedFinish')

    out_path = f"Datasets/predicted_{year}_R{round_number}_{gp_name.replace(' ','_')}.csv"
    df_sorted.to_csv(out_path, index=False)
    print(f"Predictions saved: {out_path}")
    return df_sorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--seasons', nargs='+', type=int, default=[2021,2022,2023,2024])
    args = parser.parse_args()

    # Detect GP for target round
    _, gp_name = fetch_qualifying_and_race(args.year, args.round)
    if not gp_name:
        exit()
    print(f"Target GP: {gp_name}")

    dataset = build_circuit_training_data(args.seasons, gp_name)
    if dataset.empty:
        print("No historical data found for this GP.")
        exit()

    model = train_lightgbm(dataset)
    predict_race(args.year, args.round, model)

